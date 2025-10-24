"""
視覺化模組 - 在影片幀上繪製骨架、角度、計數等資訊
"""
import cv2
import numpy as np
import mediapipe as mp


class PoseVisualizer:
    """姿態視覺化工具"""

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

    def draw_landmarks(self, frame: np.ndarray, pose_landmarks) -> np.ndarray:
        """
        在畫面上繪製 MediaPipe 骨架

        Args:
            frame: 原始影像
            pose_landmarks: MediaPipe pose landmarks

        Returns:
            繪製後的影像
        """
        if pose_landmarks is None:
            return frame

        # 自定義繪製樣式以減少閃爍
        landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=3,
            circle_radius=3
        )
        connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=3,
            circle_radius=2
        )

        # 繪製骨架連接線
        self.mp_drawing.draw_landmarks(
            frame,
            pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=landmark_drawing_spec,
            connection_drawing_spec=connection_drawing_spec
        )

        return frame

    def draw_angle(self, frame: np.ndarray, angle: float, point_b: tuple,
                   color=(255, 255, 0), thickness=2) -> np.ndarray:
        """
        在關節點旁繪製角度數值

        Args:
            frame: 影像
            angle: 角度值
            point_b: 關節中心點座標 (x, y)，範圍 0-1
            color: 文字顏色
            thickness: 線條粗細

        Returns:
            繪製後的影像
        """
        h, w, _ = frame.shape

        # 將正規化座標轉換為像素座標
        x = int(point_b[0] * w)
        y = int(point_b[1] * h)

        # 繪製角度文字（帶黑色輪廓以減少閃爍）
        text = f"{int(angle)}"
        # 黑色輪廓
        cv2.putText(
            frame,
            text,
            (x - 30, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness + 2,
            cv2.LINE_AA
        )
        # 彩色文字
        cv2.putText(
            frame,
            text,
            (x - 30, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            thickness,
            cv2.LINE_AA
        )

        # 繪製小圓圈標記關節（帶黑色邊框）
        cv2.circle(frame, (x, y), 10, (0, 0, 0), -1)
        cv2.circle(frame, (x, y), 8, color, -1)

        return frame

    def draw_counter_info(self, frame: np.ndarray, exercise_name: str,
                         count: int, stage: str, confidence: float = None) -> np.ndarray:
        """
        在畫面上繪製計數資訊面板

        Args:
            frame: 影像
            exercise_name: 運動名稱
            count: 計數
            stage: 階段 ("up" 或 "down")
            confidence: 預測信心度（可選）

        Returns:
            繪製後的影像
        """
        h, w, _ = frame.shape

        # 繪製半透明背景框（使用更高的不透明度以減少閃爍）
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 180), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

        # 運動名稱（使用更粗的線條和輪廓來減少閃爍）
        text = f"Exercise: {exercise_name}"
        # 先繪製黑色輪廓
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )
        # 再繪製白色文字
        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # 計數
        text = f"Count: {count}"
        cv2.putText(
            frame,
            text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0),
            5,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            3,
            cv2.LINE_AA
        )

        # 階段
        stage_text = "UP" if stage == "up" else "DOWN" if stage == "down" else "READY"
        stage_color = (0, 255, 0) if stage == "up" else (0, 165, 255) if stage == "down" else (200, 200, 200)
        text = f"Stage: {stage_text}"
        cv2.putText(
            frame,
            text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            4,
            cv2.LINE_AA
        )
        cv2.putText(
            frame,
            text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            stage_color,
            2,
            cv2.LINE_AA
        )

        # 信心度（如果有）
        if confidence is not None:
            text = f"Confidence: {confidence:.1f}%"
            cv2.putText(
                frame,
                text,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                4,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                text,
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

        return frame

    def draw_prediction_bar(self, frame: np.ndarray, class_names: list,
                           probabilities: np.ndarray) -> np.ndarray:
        """
        在畫面右側繪製預測機率條

        Args:
            frame: 影像
            class_names: 類別名稱列表
            probabilities: 各類別機率 (5,)

        Returns:
            繪製後的影像
        """
        h, w, _ = frame.shape

        # 從右側繪製
        x_start = w - 350
        y_start = 20
        bar_width = 300
        bar_height = 30
        spacing = 10

        for i, (name, prob) in enumerate(zip(class_names, probabilities)):
            y = y_start + i * (bar_height + spacing)

            # 繪製背景條
            cv2.rectangle(frame, (x_start, y), (x_start + bar_width, y + bar_height),
                         (50, 50, 50), -1)

            # 繪製機率條
            filled_width = int(bar_width * prob)
            color = (0, 255, 0) if i == np.argmax(probabilities) else (100, 100, 255)
            cv2.rectangle(frame, (x_start, y), (x_start + filled_width, y + bar_height),
                         color, -1)

            # 繪製文字
            text = f"{name}: {prob*100:.1f}%"
            cv2.putText(
                frame,
                text,
                (x_start + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        return frame

    def draw_instructions(self, frame: np.ndarray, message: str) -> np.ndarray:
        """
        在畫面底部繪製指示訊息

        Args:
            frame: 影像
            message: 訊息文字

        Returns:
            繪製後的影像
        """
        h, w, _ = frame.shape

        # 繪製半透明背景
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 60), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        # 繪製訊息
        cv2.putText(
            frame,
            message,
            (20, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return frame
