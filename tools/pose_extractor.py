import cv2
import numpy as np
from ultralytics import YOLO

class PoseExtractor:
    """使用 YOLOv8 Pose 從影片提取姿勢關節點座標"""

    def __init__(self):
        # 載入 YOLOv8 姿勢估計模型
        self.model = YOLO('yolov8n-pose.pt')

        # YOLOv8 輸出 17 個關節點，需要映射到 33 個點的格式
        # MediaPipe 格式: 33 點 | YOLOv8 格式: 17 點 (COCO)
        self.keypoint_mapping = self._create_keypoint_mapping()

    def _create_keypoint_mapping(self):
        """建立 YOLOv8 (17點) 到 MediaPipe (33點) 的映射"""
        # YOLOv8 COCO 格式的 17 個關節點:
        # 0:鼻子 1:左眼 2:右眼 3:左耳 4:右耳 5:左肩 6:右肩
        # 7:左肘 8:右肘 9:左腕 10:右腕 11:左髖 12:右髖
        # 13:左膝 14:右膝 15:左踝 16:右踝

        # 映射到 MediaPipe 的重要索引
        return {
            0: 0,   # 鼻子
            5: 11,  # 左肩
            6: 12,  # 右肩
            7: 13,  # 左肘
            8: 14,  # 右肘
            9: 15,  # 左腕
            10: 16, # 右腕
            11: 23, # 左髖
            12: 24, # 右髖
            13: 25, # 左膝
            14: 26, # 右膝
            15: 27, # 左踝
            16: 28, # 右踝
        }

    def extract_landmarks(self, frame):
        """
        從單一幀提取關節點座標
        Returns: shape (33, 4) 的陣列，包含 [x, y, z, visibility]
        """
        results = self.model(frame, verbose=False)

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data.cpu().numpy()

            if len(keypoints) > 0:
                kp = keypoints[0]  # 取第一個人的關節點

                # 建立 33x4 的空陣列（模擬 MediaPipe 格式）
                landmarks = np.zeros((33, 4))

                # 正規化座標到 [0, 1] 範圍
                h, w = frame.shape[:2]

                for yolo_idx, mp_idx in self.keypoint_mapping.items():
                    if yolo_idx < len(kp):
                        x, y, conf = kp[yolo_idx]
                        landmarks[mp_idx] = [x/w, y/h, 0, conf]  # z 設為 0

                return landmarks, kp

        return None, None

    def process_video(self, video_path, max_frames=None, skip_frames=0):
        """
        處理完整影片並提取所有幀的姿勢資料

        參數:
            video_path: 影片路徑
            max_frames: 最大處理幀數（None = 處理全部）
            skip_frames: 跳幀數（0 = 不跳幀，1 = 每隔1幀取1幀）

        Returns: shape (n_frames, 33, 4) 的陣列
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
        """在影片幀上繪製骨架"""
        if pose_landmarks is not None and len(pose_landmarks) > 0:
            h, w = frame.shape[:2]

            # COCO 骨架連接定義
            skeleton = [
                [5, 7], [7, 9],    # 左臂
                [6, 8], [8, 10],   # 右臂
                [5, 6],            # 肩膀
                [5, 11], [6, 12],  # 軀幹上部
                [11, 12],          # 骨盆
                [11, 13], [13, 15],# 左腿
                [12, 14], [14, 16] # 右腿
            ]

            # 繪製關節點
            for i, (x, y, conf) in enumerate(pose_landmarks):
                if conf > 0.5:  # 只繪製信心度高的點
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

            # 繪製骨架連接
            for connection in skeleton:
                pt1_idx, pt2_idx = connection
                if pt1_idx < len(pose_landmarks) and pt2_idx < len(pose_landmarks):
                    pt1 = pose_landmarks[pt1_idx]
                    pt2 = pose_landmarks[pt2_idx]

                    if pt1[2] > 0.5 and pt2[2] > 0.5:  # 信心度檢查
                        cv2.line(frame,
                                (int(pt1[0]), int(pt1[1])),
                                (int(pt2[0]), int(pt2[1])),
                                (0, 0, 255), 2)

        return frame
