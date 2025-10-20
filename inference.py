import torch
import cv2
import numpy as np
from test import BiLSTMAttention
from pose_extractor import PoseExtractor
from feature_utils import landmarks_to_features, prepare_sequence, normalize_features

class ExerciseClassifier:
    """運動動作分類系統"""

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

        # 姿勢提取器
        self.pose_extractor = PoseExtractor()

        # 類別名稱（符合立偉的模型訓練資料）
        self.class_names = [
            "槓鈴二頭彎舉 (Barbell Biceps Curl)",
            "錘式彎舉 (Hammer Curl)",
            "伏地挺身 (Push-up)",
            "肩上推舉 (Shoulder Press)",
            "深蹲 (Squat)"
        ]

    def predict_video(self, video_path, target_length=30):
        """
        對影片進行分類預測

        Returns:
            predicted_class: 預測類別索引
            confidence: 信心度
            probabilities: 所有類別的機率分布
        """
        print(f"正在處理影片: {video_path}")

        # 1. 提取姿勢關節點
        landmarks_sequence = self.pose_extractor.process_video(video_path)
        if landmarks_sequence is None or len(landmarks_sequence) == 0:
            raise ValueError("無法從影片中檢測到姿勢")

        print(f"檢測到 {len(landmarks_sequence)} 幀姿勢資料")

        # 2. 轉換為特徵
        features = landmarks_to_features(landmarks_sequence)
        features = normalize_features(features)
        features = prepare_sequence(features, target_length=target_length)

        # 3. 模型推論
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0).to(self.device)  # [1, seq_len, 102]
            output = self.model(x)  # [1, num_classes]
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class]

        return predicted_class, confidence, probabilities

    def predict_and_visualize(self, video_path, output_path=None, target_length=30):
        """
        對影片進行分類並產生帶有視覺化結果的輸出影片

        Args:
            video_path: 輸入影片路徑
            output_path: 輸出影片路徑（若為 None 則不儲存）
            target_length: 序列長度
        """
        # 先進行預測
        predicted_class, confidence, probabilities = self.predict_video(video_path, target_length)

        print(f"\n預測結果: {self.class_names[predicted_class]}")
        print(f"信心度: {confidence:.2%}")
        print("\n所有類別機率:")
        for i, prob in enumerate(probabilities):
            print(f"  {self.class_names[i]}: {prob:.2%}")

        # 產生視覺化影片
        cap = cv2.VideoCapture(video_path)
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 繪製姿勢骨架
            landmarks, pose_landmarks = self.pose_extractor.extract_landmarks(frame)
            if pose_landmarks:
                frame = self.pose_extractor.draw_landmarks(frame, pose_landmarks)

            # 繪製預測結果
            label_text = f"動作: {self.class_names[predicted_class]}"
            conf_text = f"信心度: {confidence:.2%}"

            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, conf_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if output_path:
                out.write(frame)

            frame_count += 1

        cap.release()
        if output_path:
            out.release()
            print(f"\n輸出影片已儲存至: {output_path}")

        return predicted_class, confidence, probabilities


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("使用方式: python inference.py <影片路徑> [輸出路徑]")
        print("範例: python inference.py squat.mp4 output.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    classifier = ExerciseClassifier()
    classifier.predict_and_visualize(video_path, output_path)
