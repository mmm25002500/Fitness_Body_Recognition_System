"""
改進版推論系統：實作立偉模型的完整推論流程
- 使用 45 幀滑動窗口（步幅 3）
- 實作影片級 Top-K 軟投票機制
- Z-score 標準化
"""

import torch
import cv2
import numpy as np
from test import BiLSTMAttention
from pose_config import get_pose_extractor
from feature_utils_v2 import (
    landmarks_to_features_v2,
    create_sliding_windows,
    z_score_normalize
)

class ExerciseClassifierV2:
    """運動動作分類系統 V2（符合立偉的實作）"""

    def __init__(self, model_path="bilstm_mix_best_pt.pth", device=None):
        # 設定運算裝置
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # 載入模型
        self.model = BiLSTMAttention(input_dim=102, hidden_dim=96, attn_dim=128, num_classes=5)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        # 姿勢提取器（自動選擇 MediaPipe 或 YOLOv8）
        self.pose_extractor = get_pose_extractor(prefer_mediapipe=True)

        # 類別名稱
        self.class_names = [
            "槓鈴二頭彎舉 (Barbell Biceps Curl)",
            "錘式彎舉 (Hammer Curl)",
            "伏地挺身 (Push-up)",
            "肩上推舉 (Shoulder Press)",
            "深蹲 (Squat)"
        ]

    def predict_video_sequence_level(self, video_path, window_size=45, stride=3):
        """
        序列級預測：對影片的所有滑動窗口進行預測

        返回:
            all_logits: 所有窗口的 logit 輸出 (n_windows, num_classes)
            all_probs: 所有窗口的機率分佈 (n_windows, num_classes)
        """
        print(f"正在處理影片: {video_path}")

        # 1. 提取姿勢關節點
        landmarks_sequence = self.pose_extractor.process_video(video_path)
        if landmarks_sequence is None or len(landmarks_sequence) == 0:
            raise ValueError("無法從影片中檢測到姿勢")

        print(f"檢測到 {len(landmarks_sequence)} 幀姿勢資料")

        # 2. 轉換為特徵（角度 + 座標混合，102 維）
        features = landmarks_to_features_v2(landmarks_sequence)

        # 3. Z-score 標準化
        features = z_score_normalize(features)

        # 4. 建立滑動窗口
        windows = create_sliding_windows(features, window_size=window_size, stride=stride)
        print(f"建立 {len(windows)} 個滑動窗口 (大小={window_size}, 步幅={stride})")

        # 5. 批次推論
        all_logits = []
        all_probs = []

        with torch.no_grad():
            for window in windows:
                x = torch.FloatTensor(window).unsqueeze(0).to(self.device)  # [1, 45, 102]
                output = self.model(x)  # [1, 5]
                probs = torch.softmax(output, dim=1)

                all_logits.append(output.cpu().numpy()[0])
                all_probs.append(probs.cpu().numpy()[0])

        return np.array(all_logits), np.array(all_probs)

    def video_level_prediction_topk(self, all_logits, all_probs, k=5, nms_threshold=0.6):
        """
        影片級預測：使用改進的投票機制

        參數:
            all_logits: 所有窗口的 logit (n_windows, num_classes)
            all_probs: 所有窗口的機率 (n_windows, num_classes)
            k: 選擇前 K 個最有信心的窗口
            nms_threshold: 時間軸非極大值抑制的閾值

        返回:
            predicted_class: 預測類別
            confidence: 信心度
            final_probs: 最終機率分佈
        """
        n_windows = len(all_logits)

        # 1. 計算每個窗口的最大機率（信心度）
        max_probs = np.max(all_probs, axis=1)
        window_predictions = np.argmax(all_probs, axis=1)

        print(f"\n窗口預測詳情 (共 {n_windows} 個窗口):")
        for i, (pred_class, conf) in enumerate(zip(window_predictions, max_probs)):
            print(f"  窗口 {i}: {self.class_names[pred_class]} (信心度: {conf:.2%})")

        # 2. 對於少量窗口（<5），使用加權平均所有窗口
        if n_windows < 5:
            print(f"\n窗口數量較少 ({n_windows})，使用所有窗口的加權平均")
            # 使用信心度作為權重
            weights = max_probs / np.sum(max_probs)
            final_probs = np.average(all_probs, axis=0, weights=weights)

        else:
            # 3. 對於較多窗口，使用 Top-K + NMS
            # 時間軸非極大值抑制
            max_logits = np.max(all_logits, axis=1)
            selected_indices = self._temporal_nms(max_logits, threshold=nms_threshold)

            print(f"\nNMS 後保留 {len(selected_indices)} 個窗口")

            # 從篩選後的窗口中選擇 Top-K
            selected_probs = max_probs[selected_indices]
            k = min(k, len(selected_indices))
            topk_in_selected = np.argsort(selected_probs)[-k:]
            topk_indices = selected_indices[topk_in_selected]

            print(f"選擇 Top-{k} 窗口: {topk_indices}")

            # 加權平均
            topk_max_probs = max_probs[topk_indices]
            weights = topk_max_probs / np.sum(topk_max_probs)
            final_probs = np.average(all_probs[topk_indices], axis=0, weights=weights)

        # 5. 最終預測
        predicted_class = np.argmax(final_probs)
        confidence = final_probs[predicted_class]

        return predicted_class, confidence, final_probs

    def _temporal_nms(self, scores, threshold=0.6):
        """
        時間軸非極大值抑制

        參數:
            scores: 每個窗口的信心度分數
            threshold: 相對分數閾值（保留分數 > threshold × max_score 的窗口）

        返回:
            selected_indices: 保留的窗口索引
        """
        max_score = np.max(scores)
        threshold_value = threshold * max_score

        # 保留分數高於閾值的窗口
        selected = np.where(scores >= threshold_value)[0]

        # 進一步抑制連續窗口（只保留局部最大值）
        final_selected = []
        i = 0
        while i < len(selected):
            # 找到當前位置的局部峰值
            current_idx = selected[i]
            local_peak = current_idx

            # 檢查連續窗口
            j = i + 1
            while j < len(selected) and selected[j] - selected[j-1] <= 3:  # 連續窗口（步幅3）
                if scores[selected[j]] > scores[local_peak]:
                    local_peak = selected[j]
                j += 1

            final_selected.append(local_peak)
            i = j

        return np.array(final_selected)

    def predict_video(self, video_path, window_size=45, stride=3, topk=None, nms_threshold=0.5):
        """
        完整的影片預測流程

        參數:
            video_path: 影片路徑
            window_size: 窗口大小（預設 45）
            stride: 步幅（預設 3）
            topk: 選擇前 K 個窗口（None = 自動調整為窗口數的 1/3）
            nms_threshold: NMS 閾值（預設 0.5）

        返回:
            predicted_class: 預測類別
            confidence: 信心度
            final_probs: 機率分佈
        """
        # 序列級預測
        all_logits, all_probs = self.predict_video_sequence_level(
            video_path, window_size=window_size, stride=stride
        )

        # 自動調整 topk
        if topk is None:
            topk = max(3, len(all_logits) // 3)  # 至少3個，最多1/3窗口數

        topk = min(topk, len(all_logits))  # 不超過總窗口數

        print(f"使用 Top-{topk} 窗口進行投票（NMS閾值={nms_threshold}）")

        # 影片級整合
        predicted_class, confidence, final_probs = self.video_level_prediction_topk(
            all_logits, all_probs, k=topk, nms_threshold=nms_threshold
        )

        return predicted_class, confidence, final_probs

    def predict_and_visualize(self, video_path, output_path=None, window_size=45, stride=3, topk=5):
        """產生帶視覺化的預測結果"""
        # 進行預測
        predicted_class, confidence, probabilities = self.predict_video(
            video_path, window_size=window_size, stride=stride, topk=topk
        )

        print(f"\n預測結果: {self.class_names[predicted_class]}")
        print(f"信心度: {confidence:.2%}")
        print("\n所有類別機率:")
        for i, prob in enumerate(probabilities):
            print(f"  {self.class_names[i]}: {prob:.2%}")

        # 產生視覺化影片
        if output_path:
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 繪製姿勢骨架
                landmarks, pose_landmarks = self.pose_extractor.extract_landmarks(frame)
                if pose_landmarks is not None:
                    frame = self.pose_extractor.draw_landmarks(frame, pose_landmarks)

                # 繪製預測結果
                label_text = f"動作: {self.class_names[predicted_class]}"
                conf_text = f"信心度: {confidence:.2%}"

                cv2.rectangle(frame, (10, 10), (500, 100), (0, 0, 0), -1)
                cv2.putText(frame, label_text, (20, 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, conf_text, (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                out.write(frame)

            cap.release()
            out.release()
            print(f"\n輸出影片已儲存至: {output_path}")

        return predicted_class, confidence, probabilities


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方式: python inference_v2.py <影片路徑> [輸出路徑]")
        print("範例: python inference_v2.py squat.mp4 output.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    classifier = ExerciseClassifierV2()
    classifier.predict_and_visualize(video_path, output_path)
