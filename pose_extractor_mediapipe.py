"""
MediaPipe 版本的姿勢提取器
要求: Python 3.11 或更早版本
套件: mediapipe>=0.10.0

使用方式:
    pip install mediapipe==0.10.14
"""

import cv2
import numpy as np

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("警告: MediaPipe 未安裝，請使用 Python 3.11 並執行: pip install mediapipe")

class PoseExtractorMediaPipe:
    """使用 MediaPipe 從影片提取完整的 33 個關節點座標"""

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError(
                "MediaPipe 不可用。\n"
                "請確保:\n"
                "1. 使用 Python 3.11 或更早版本\n"
                "2. 執行: pip install mediapipe>=0.10.0"
            )

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # 初始化 MediaPipe Pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,  # 0: Lite, 1: Full, 2: Heavy
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_landmarks(self, frame):
        """
        從單一幀提取 MediaPipe 的 33 個關節點座標

        Returns:
            landmarks: shape (33, 4) 的陣列，包含 [x, y, z, visibility]
            pose_landmarks: MediaPipe 的原始 pose_landmarks 物件（用於繪圖）
        """
        # 轉換為 RGB（MediaPipe 需要 RGB 格式）
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 處理影像
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks:
            # 提取所有 33 個關節點
            landmarks = []
            for lm in results.pose_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z, lm.visibility])

            return np.array(landmarks, dtype=np.float32), results.pose_landmarks

        return None, None

    def process_video(self, video_path, max_frames=None, skip_frames=0):
        """
        處理完整影片並提取所有幀的姿勢資料

        參數:
            video_path: 影片路徑
            max_frames: 最大處理幀數（None = 處理全部）
            skip_frames: 跳幀數（0 = 不跳幀，1 = 每隔1幀取1幀）

        Returns:
            shape (n_frames, 33, 4) 的陣列
        """
        cap = cv2.VideoCapture(video_path)
        all_landmarks = []
        frame_count = 0
        processed_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 跳幀邏輯
            if skip_frames > 0 and frame_count % (skip_frames + 1) != 0:
                frame_count += 1
                continue

            landmarks, _ = self.extract_landmarks(frame)
            if landmarks is not None:
                all_landmarks.append(landmarks)
                processed_count += 1

            frame_count += 1
            if max_frames and processed_count >= max_frames:
                break

        cap.release()
        print(f"處理了 {frame_count} 幀，成功提取 {len(all_landmarks)} 幀姿勢")
        return np.array(all_landmarks) if all_landmarks else None

    def draw_landmarks(self, frame, pose_landmarks):
        """
        在影片幀上繪製 MediaPipe 骨架

        參數:
            frame: 影像幀
            pose_landmarks: MediaPipe 的 pose_landmarks 物件
        """
        if pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
        return frame

    def get_landmark_names(self):
        """返回 MediaPipe 33 個關節點的名稱"""
        return [
            "鼻子 (NOSE)",
            "左眼內側 (LEFT_EYE_INNER)",
            "左眼 (LEFT_EYE)",
            "左眼外側 (LEFT_EYE_OUTER)",
            "右眼內側 (RIGHT_EYE_INNER)",
            "右眼 (RIGHT_EYE)",
            "右眼外側 (RIGHT_EYE_OUTER)",
            "左耳 (LEFT_EAR)",
            "右耳 (RIGHT_EAR)",
            "嘴巴左側 (MOUTH_LEFT)",
            "嘴巴右側 (MOUTH_RIGHT)",
            "左肩 (LEFT_SHOULDER)",
            "右肩 (RIGHT_SHOULDER)",
            "左肘 (LEFT_ELBOW)",
            "右肘 (RIGHT_ELBOW)",
            "左腕 (LEFT_WRIST)",
            "右腕 (RIGHT_WRIST)",
            "左小指 (LEFT_PINKY)",
            "右小指 (RIGHT_PINKY)",
            "左食指 (LEFT_INDEX)",
            "右食指 (RIGHT_INDEX)",
            "左拇指 (LEFT_THUMB)",
            "右拇指 (RIGHT_THUMB)",
            "左髖 (LEFT_HIP)",
            "右髖 (RIGHT_HIP)",
            "左膝 (LEFT_KNEE)",
            "右膝 (RIGHT_KNEE)",
            "左踝 (LEFT_ANKLE)",
            "右踝 (RIGHT_ANKLE)",
            "左腳跟 (LEFT_HEEL)",
            "右腳跟 (RIGHT_HEEL)",
            "左腳尖 (LEFT_FOOT_INDEX)",
            "右腳尖 (RIGHT_FOOT_INDEX)"
        ]

    def __del__(self):
        """清理資源"""
        if hasattr(self, 'pose'):
            self.pose.close()


# 測試代碼
if __name__ == "__main__":
    import sys

    if not MEDIAPIPE_AVAILABLE:
        print("MediaPipe 未安裝，無法執行測試")
        sys.exit(1)

    print("MediaPipe 姿勢提取器測試\n")

    extractor = PoseExtractorMediaPipe()

    # 測試單幀提取
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        print(f"測試影片: {video_path}")

        # 提取前 10 幀
        landmarks = extractor.process_video(video_path, max_frames=10)

        if landmarks is not None:
            print(f"\n提取成功！")
            print(f"形狀: {landmarks.shape}")
            print(f"\n第一幀的前 5 個關節點:")
            print(landmarks[0][:5])

            print(f"\n關節點名稱:")
            for i, name in enumerate(extractor.get_landmark_names()[:5]):
                print(f"  {i}: {name}")
        else:
            print("提取失敗")
    else:
        print("使用方式: python pose_extractor_mediapipe.py <影片路徑>")
