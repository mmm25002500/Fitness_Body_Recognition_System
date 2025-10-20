"""
Fitness AI Trainer - Streamlit Web 應用
整合運動辨識、計數、視覺化功能
"""
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from pathlib import Path

# 確保使用 MediaPipe
os.environ["PREFER_MEDIAPIPE"] = "1"

from pose_config import get_pose_extractor
from test import BiLSTMAttention
from exercise_counter import RepetitionCounter
from visualization import PoseVisualizer
import torch


class FitnessAITrainer:
    """健身 AI 訓練系統"""

    def __init__(self, model_path="bilstm_mix_best_pt.pth"):
        """初始化系統"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 載入模型
        self.model = BiLSTMAttention(
            input_dim=102,
            hidden_dim=96,
            attn_dim=128,
            num_classes=5
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        # 初始化模組
        self.pose_extractor = get_pose_extractor(prefer_mediapipe=True)
        self.visualizer = PoseVisualizer()

        # 類別名稱
        self.class_names = [
            "槓鈴二頭彎舉",
            "錘式彎舉",
            "伏地挺身",
            "肩上推舉",
            "深蹲"
        ]

        # 滑動窗口
        self.window_size = 45
        self.feature_buffer = []
        self.current_exercise = None
        self.counter = None

    def extract_features(self, landmarks):
        """從 landmarks 提取 102 維特徵"""
        if landmarks is None:
            return None

        features = []

        # 1. 8個關節角度
        angles = self._calculate_joint_angles(landmarks)
        features.extend(angles)

        # 2. 17個關鍵點的 3D 座標 (51 個值)
        key_points_idx = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
        for idx in key_points_idx:
            features.extend(landmarks[idx][:3])  # x, y, z

        # 3. 幾何特徵 (43 個值)
        geometric = self._calculate_geometric_features(landmarks)
        features.extend(geometric)

        return np.array(features, dtype=np.float32)

    def _calculate_joint_angles(self, landmarks):
        """計算8個關節角度"""
        def angle_3d(a, b, c):
            ba = a - b
            bc = c - b
            cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
            return np.arccos(np.clip(cosine, -1.0, 1.0))

        angles = []
        # 左右手肘、肩膀、膝蓋、髖部
        joints = [
            (11, 13, 15), (12, 14, 16),  # 左右手肘
            (13, 11, 23), (14, 12, 24),  # 左右肩膀
            (23, 25, 27), (24, 26, 28),  # 左右膝蓋
            (11, 23, 25), (12, 24, 26),  # 左右髖部
        ]

        for a_idx, b_idx, c_idx in joints:
            a = landmarks[a_idx][:3]
            b = landmarks[b_idx][:3]
            c = landmarks[c_idx][:3]
            angles.append(angle_3d(a, b, c))

        return angles

    def _calculate_geometric_features(self, landmarks):
        """計算幾何特徵"""
        features = []

        # 身體中心點
        center = (landmarks[11][:3] + landmarks[12][:3]) / 2

        # 計算相對於中心的距離和角度
        key_points = [0, 15, 16, 23, 24, 27, 28]
        for idx in key_points:
            diff = landmarks[idx][:3] - center
            dist = np.linalg.norm(diff)
            features.append(dist)
            features.extend(diff)  # x, y, z 差值

        # 肢體長度比例
        torso_length = np.linalg.norm(landmarks[11][:3] - landmarks[23][:3])
        left_arm = np.linalg.norm(landmarks[11][:3] - landmarks[15][:3])
        right_arm = np.linalg.norm(landmarks[12][:3] - landmarks[16][:3])
        left_leg = np.linalg.norm(landmarks[23][:3] - landmarks[27][:3])
        right_leg = np.linalg.norm(landmarks[24][:3] - landmarks[28][:3])

        features.extend([torso_length, left_arm, right_arm, left_leg, right_leg])

        # 填充到 43 維
        while len(features) < 43:
            features.append(0.0)

        return features[:43]

    def predict_exercise(self, features_sequence):
        """預測運動類型"""
        if len(features_sequence) < self.window_size:
            return None, None

        # 取最後 45 幀
        sequence = np.array(features_sequence[-self.window_size:])

        # Z-score 正規化
        mean = sequence.mean(axis=0)
        std = sequence.std(axis=0) + 1e-6
        sequence = (sequence - mean) / std

        # 轉換為 tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)

        # 預測
        with torch.no_grad():
            output = self.model(sequence_tensor)
            probabilities = torch.softmax(output, dim=1).cpu().numpy()[0]

        predicted_class = np.argmax(probabilities)
        confidence = probabilities[predicted_class] * 100

        return predicted_class, probabilities

    def process_frame(self, frame, mode="automatic", debug=False):
        """
        處理單幀影像

        Args:
            frame: BGR 影像
            mode: "automatic" 自動模式，"manual" 手動模式
            debug: 是否啟用除錯模式

        Returns:
            處理後的影像
        """
        # 姿態估計
        landmarks, pose_landmarks = self.pose_extractor.extract_landmarks(frame)

        if landmarks is None:
            self.visualizer.draw_instructions(frame, "未偵測到人物，請確保全身在畫面中")
            return frame

        # 繪製骨架
        frame = self.visualizer.draw_landmarks(frame, pose_landmarks)

        # 提取特徵
        features = self.extract_features(landmarks)
        if features is not None:
            self.feature_buffer.append(features)

            # 自動模式：運動辨識
            if mode == "automatic" and len(self.feature_buffer) >= self.window_size:
                predicted_id, probabilities = self.predict_exercise(self.feature_buffer)

                if predicted_id is not None:
                    # 檢查是否需要創建或更新計數器
                    need_new_counter = False

                    if self.counter is None:
                        need_new_counter = True
                    elif self.current_exercise != predicted_id:
                        # 運動類型改變，需要新計數器
                        need_new_counter = True

                    if need_new_counter:
                        self.current_exercise = predicted_id
                        self.counter = RepetitionCounter(self.class_names[predicted_id])
                        self.counter.debug = debug

                    # 更新除錯模式
                    self.counter.debug = debug

                    # 更新計數
                    counter_result = self.counter.update(landmarks, predicted_id)

                    # 繪製資訊
                    frame = self.visualizer.draw_counter_info(
                        frame,
                        self.class_names[predicted_id],
                        counter_result["count"],
                        counter_result["stage"],
                        probabilities[predicted_id] * 100
                    )

                    # 繪製角度
                    if counter_result["angle"] is not None and counter_result["points"] is not None:
                        _, point_b, _ = counter_result["points"]
                        frame = self.visualizer.draw_angle(
                            frame,
                            counter_result["angle"],
                            point_b
                        )

                    # 繪製預測機率條
                    frame = self.visualizer.draw_prediction_bar(
                        frame,
                        self.class_names,
                        probabilities
                    )

            # 手動模式：使用者選擇運動
            elif mode == "manual" and self.current_exercise is not None:
                counter_result = self.counter.update(landmarks, self.current_exercise)

                frame = self.visualizer.draw_counter_info(
                    frame,
                    self.class_names[self.current_exercise],
                    counter_result["count"],
                    counter_result["stage"]
                )

                if counter_result["angle"] is not None and counter_result["points"] is not None:
                    _, point_b, _ = counter_result["points"]
                    frame = self.visualizer.draw_angle(
                        frame,
                        counter_result["angle"],
                        point_b
                    )

        return frame

    def reset(self):
        """重置系統狀態"""
        self.feature_buffer = []
        self.current_exercise = None
        self.counter = None


def main():
    """Streamlit 主應用"""
    st.set_page_config(
        page_title="Fitness AI Trainer",
        page_icon="💪",
        layout="wide"
    )

    st.title("💪 Fitness AI Trainer - 運動辨識與計數系統")
    st.markdown("---")

    # 初始化系統
    if "trainer" not in st.session_state:
        with st.spinner("載入 AI 模型中..."):
            st.session_state.trainer = FitnessAITrainer()
        st.success("✓ AI 模型載入完成！")

    trainer = st.session_state.trainer

    # 初始化計數器狀態（使用 session_state 保存計數器）
    if "current_counter" not in st.session_state:
        st.session_state.current_counter = None
    if "last_exercise" not in st.session_state:
        st.session_state.last_exercise = None
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = None

    # 側邊欄設定
    st.sidebar.header("⚙️ 設定")

    mode = st.sidebar.radio(
        "選擇模式",
        ["automatic", "manual"],
        format_func=lambda x: "🤖 自動模式（AI 辨識）" if x == "automatic" else "✋ 手動模式（手動選擇）"
    )

    # 除錯模式
    debug_mode = st.sidebar.checkbox("🐛 除錯模式（顯示角度資訊）", value=False)

    # 手動模式下選擇運動
    if mode == "manual":
        exercise_choice = st.sidebar.selectbox(
            "選擇運動類型",
            range(5),
            format_func=lambda x: trainer.class_names[x]
        )

        # 只在運動類型或模式改變時重新創建計數器
        if (st.session_state.last_exercise != exercise_choice or
            st.session_state.last_mode != mode or
            st.session_state.current_counter is None):

            st.session_state.last_exercise = exercise_choice
            st.session_state.last_mode = mode
            st.session_state.current_counter = RepetitionCounter(trainer.class_names[exercise_choice])
            st.session_state.current_counter.debug = debug_mode

        # 從 session_state 恢復計數器
        trainer.current_exercise = exercise_choice
        trainer.counter = st.session_state.current_counter

        # 更新除錯模式
        if trainer.counter is not None:
            trainer.counter.debug = debug_mode

    # 自動模式：準備計數器（但不指定運動類型）
    else:
        # 如果模式改變，重置計數器
        if st.session_state.last_mode != mode:
            st.session_state.last_mode = mode
            st.session_state.current_counter = None
            st.session_state.last_exercise = None

    input_type = st.sidebar.radio(
        "選擇輸入來源",
        ["video", "webcam"],
        format_func=lambda x: "📹 影片檔案" if x == "video" else "📷 即時攝影機"
    )

    # 重置按鈕
    if st.sidebar.button("🔄 重置計數器"):
        # 重置 session_state 中的計數器
        st.session_state.current_counter = None
        st.session_state.last_exercise = None
        st.session_state.last_mode = None
        trainer.reset()
        st.sidebar.success("計數器已重置！")
        st.rerun()

    # 影片上傳模式
    if input_type == "video":
        st.sidebar.markdown("---")
        st.sidebar.info("📤 請上傳運動影片（支援 mp4, avi, mov）")

        uploaded_file = st.file_uploader(
            "選擇影片檔案",
            type=["mp4", "avi", "mov"]
        )

        if uploaded_file is not None:
            # 儲存暫存檔案
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            video_path = tfile.name

            st.markdown("### 📊 分析結果")

            # 影片資訊
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            duration = total_frames / fps if fps > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("總幀數", f"{total_frames} 幀")
            col2.metric("FPS", f"{fps}")
            col3.metric("長度", f"{duration:.1f} 秒")

            # 處理影片
            stframe = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()

            frame_count = 0
            # 不要重置，保留 session_state 中的計數器
            # 只重置特徵緩衝區
            trainer.feature_buffer = []

            # 確保使用 session_state 的計數器
            if st.session_state.current_counter is not None:
                trainer.counter = st.session_state.current_counter

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 在處理前，先從 session_state 恢復計數器
                if st.session_state.current_counter is not None:
                    trainer.counter = st.session_state.current_counter

                # 處理幀
                processed_frame = trainer.process_frame(frame, mode=mode, debug=debug_mode)

                # 處理後，同步計數器回 session_state（重要！）
                if trainer.counter is not None:
                    st.session_state.current_counter = trainer.counter
                    # 同時更新運動類型記錄（自動模式）
                    if mode == "automatic" and trainer.current_exercise is not None:
                        st.session_state.last_exercise = trainer.current_exercise

                # 顯示
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)

                # 更新進度
                frame_count += 1
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                status_text.text(f"處理中... {frame_count}/{total_frames} 幀 ({progress*100:.1f}%)")

            cap.release()
            os.unlink(video_path)

            st.success("✓ 影片處理完成！")

            # 顯示最終統計
            if st.session_state.current_counter is not None:
                st.markdown("### 📈 統計結果")
                final_count = st.session_state.current_counter.count
                st.metric("總次數", final_count, delta=None)

                # 顯示詳細資訊
                col1, col2, col3 = st.columns(3)
                col1.metric("運動類型", trainer.class_names[st.session_state.last_exercise])
                col2.metric("總次數", final_count)
                col3.metric("處理幀數", frame_count)

    # 即時攝影機模式
    else:
        st.info("📷 即時攝影機模式")
        st.warning("⚠️ 此功能需要在本地運行 Streamlit 才能使用攝影機")
        st.markdown("""
        請在終端機執行：
        ```bash
        source venv_mediapipe/bin/activate
        streamlit run app.py
        ```
        然後在瀏覽器中允許攝影機權限。
        """)

        run_webcam = st.checkbox("啟動攝影機")

        if run_webcam:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            stop_button = st.button("停止")

            trainer.reset()

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("無法讀取攝影機")
                    break

                # 處理幀
                processed_frame = trainer.process_frame(frame, mode=mode, debug=debug_mode)

                # 顯示
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe.image(processed_frame_rgb, channels="RGB", use_column_width=True)

            cap.release()

    # 說明文件
    with st.expander("📖 使用說明"):
        st.markdown("""
        ## 功能說明

        ### 🤖 自動模式
        - AI 自動辨識運動類型
        - 自動計算次數
        - 顯示預測信心度和各類別機率

        ### ✋ 手動模式
        - 手動選擇運動類型
        - 自動計算次數
        - 適合專注訓練特定動作

        ### 支援的運動
        1. 槓鈴二頭彎舉 (Barbell Biceps Curl)
        2. 錘式彎舉 (Hammer Curl)
        3. 伏地挺身 (Push-up)
        4. 肩上推舉 (Shoulder Press)
        5. 深蹲 (Squat)

        ### 使用技巧
        - 確保全身在畫面中
        - 保持良好的光線
        - 動作完整清晰
        - 避免快速移動造成模糊
        """)


if __name__ == "__main__":
    main()
