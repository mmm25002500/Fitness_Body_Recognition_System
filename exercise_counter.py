"""
運動計數模組 - 使用角度閾值檢測自動計算運動次數
支援 5 種運動類型的次數統計
"""
import numpy as np
from typing import Optional, Tuple


class RepetitionCounter:
    """運動次數計數器"""

    def __init__(self, exercise_type: str):
        """
        初始化計數器

        Args:
            exercise_type: 運動類型
                0: 槓鈴二頭彎舉
                1: 錘式彎舉
                2: 伏地挺身
                3: 肩上推舉
                4: 深蹲
        """
        self.exercise_type = exercise_type
        self.exercise_names = {
            0: "槓鈴二頭彎舉",
            1: "錘式彎舉",
            2: "伏地挺身",
            3: "肩上推舉",
            4: "深蹲"
        }

        # 計數器狀態
        self.count = 0
        self.stage = None  # "up" 或 "down"

        # 每種運動的角度閾值配置（自動適應不同動作幅度）
        # 統一邏輯：DOWN階段 → UP階段 = 完成一次
        self.thresholds = {
            # 彎舉類：手臂伸直(大角度)=DOWN，彎曲(小角度)=UP
            0: {"down_angle": 150, "up_angle": 50, "joint": "elbow", "count_on": "up"},      # 槓鈴二頭彎舉
            1: {"down_angle": 150, "up_angle": 50, "joint": "elbow", "count_on": "up"},      # 錘式彎舉

            # 伏地挺身：手臂伸直(大角度)=UP，彎曲(小角度)=DOWN
            2: {"down_angle": 70, "up_angle": 150, "joint": "elbow", "count_on": "up"},

            # 肩推：手臂伸直向上(大角度)=UP，收回(小角度)=DOWN
            # 放寬閾值以適應不同動作幅度
            3: {"down_angle": 50, "up_angle": 130, "joint": "elbow", "count_on": "up"},

            # 深蹲：站立(大角度)=UP，蹲下(小角度)=DOWN
            4: {"down_angle": 170, "up_angle": 80, "joint": "knee", "count_on": "up"}
        }

        # 動態閾值調整（如果偵測到角度範圍太小）
        self.angle_history = []
        self.adaptive_threshold = True

        # 除錯模式
        self.debug = False

    def calculate_angle(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        計算三點之間的角度（以 b 為中心點）
        使用向量夾角公式

        Args:
            a: 第一個點 [x, y]
            b: 中心點 [x, y]
            c: 第三個點 [x, y]

        Returns:
            角度（度數，0-180）
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # 計算向量
        ba = a - b
        bc = c - b

        # 使用餘弦定理計算角度
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        # 限制在 [-1, 1] 範圍內避免數值誤差
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def get_joint_points(self, landmarks: np.ndarray, exercise_id: int) -> Optional[Tuple]:
        """
        根據運動類型獲取關鍵關節點

        Args:
            landmarks: MediaPipe landmarks (33, 4) - [x, y, z, visibility]
            exercise_id: 運動類型 ID

        Returns:
            (關節角度, 點A, 點B, 點C) 或 None
        """
        # MediaPipe Pose Landmark 索引
        # 參考: https://google.github.io/mediapipe/solutions/pose.html
        NOSE = 0
        LEFT_SHOULDER = 11
        RIGHT_SHOULDER = 12
        LEFT_ELBOW = 13
        RIGHT_ELBOW = 14
        LEFT_WRIST = 15
        RIGHT_WRIST = 16
        LEFT_HIP = 23
        RIGHT_HIP = 24
        LEFT_KNEE = 25
        RIGHT_KNEE = 26
        LEFT_ANKLE = 27
        RIGHT_ANKLE = 28

        config = self.thresholds.get(exercise_id)
        if not config:
            return None

        try:
            if config["joint"] == "elbow":
                # 計算右手肘角度（優先使用右手）
                if landmarks[RIGHT_SHOULDER][3] > 0.5 and \
                   landmarks[RIGHT_ELBOW][3] > 0.5 and \
                   landmarks[RIGHT_WRIST][3] > 0.5:

                    shoulder = landmarks[RIGHT_SHOULDER][:2]
                    elbow = landmarks[RIGHT_ELBOW][:2]
                    wrist = landmarks[RIGHT_WRIST][:2]

                    angle = self.calculate_angle(shoulder, elbow, wrist)
                    return angle, shoulder, elbow, wrist

                # 備用：使用左手
                elif landmarks[LEFT_SHOULDER][3] > 0.5 and \
                     landmarks[LEFT_ELBOW][3] > 0.5 and \
                     landmarks[LEFT_WRIST][3] > 0.5:

                    shoulder = landmarks[LEFT_SHOULDER][:2]
                    elbow = landmarks[LEFT_ELBOW][:2]
                    wrist = landmarks[LEFT_WRIST][:2]

                    angle = self.calculate_angle(shoulder, elbow, wrist)
                    return angle, shoulder, elbow, wrist

            elif config["joint"] == "knee":
                # 計算右膝蓋角度（優先使用右腿）
                if landmarks[RIGHT_HIP][3] > 0.5 and \
                   landmarks[RIGHT_KNEE][3] > 0.5 and \
                   landmarks[RIGHT_ANKLE][3] > 0.5:

                    hip = landmarks[RIGHT_HIP][:2]
                    knee = landmarks[RIGHT_KNEE][:2]
                    ankle = landmarks[RIGHT_ANKLE][:2]

                    angle = self.calculate_angle(hip, knee, ankle)
                    return angle, hip, knee, ankle

                # 備用：使用左腿
                elif landmarks[LEFT_HIP][3] > 0.5 and \
                     landmarks[LEFT_KNEE][3] > 0.5 and \
                     landmarks[LEFT_ANKLE][3] > 0.5:

                    hip = landmarks[LEFT_HIP][:2]
                    knee = landmarks[LEFT_KNEE][:2]
                    ankle = landmarks[LEFT_ANKLE][:2]

                    angle = self.calculate_angle(hip, knee, ankle)
                    return angle, hip, knee, ankle

        except Exception as e:
            print(f"計算角度時發生錯誤: {e}")
            return None

        return None

    def update(self, landmarks: np.ndarray, exercise_id: int) -> dict:
        """
        更新計數器狀態

        Args:
            landmarks: MediaPipe landmarks (33, 4)
            exercise_id: 運動類型 ID

        Returns:
            包含計數、角度、狀態的字典
        """
        result = {
            "count": self.count,
            "stage": self.stage,
            "angle": None,
            "points": None
        }

        joint_data = self.get_joint_points(landmarks, exercise_id)
        if joint_data is None:
            return result

        angle, point_a, point_b, point_c = joint_data
        result["angle"] = angle
        result["points"] = (point_a, point_b, point_c)

        config = self.thresholds[exercise_id]
        down_threshold = config["down_angle"]
        up_threshold = config["up_angle"]

        # 動態閾值適應（記錄角度歷史）
        self.angle_history.append(angle)
        if len(self.angle_history) > 100:
            self.angle_history.pop(0)

        # 如果角度範圍太小，自動調整閾值
        if len(self.angle_history) >= 50 and self.adaptive_threshold:
            min_angle = min(self.angle_history)
            max_angle = max(self.angle_history)
            angle_range = max_angle - min_angle

            # 如果角度變化範圍小於 20°，使用相對閾值
            if angle_range < 20 and angle_range > 5:
                mid_angle = (min_angle + max_angle) / 2
                if exercise_id in [0, 1]:  # 彎舉
                    down_threshold = max_angle - 2
                    up_threshold = min_angle + 2
                elif exercise_id == 2:  # 伏地挺身
                    down_threshold = min_angle + 2
                    up_threshold = max_angle - 2
                elif exercise_id == 3:  # 肩推
                    down_threshold = min_angle + 2
                    up_threshold = max_angle - 2
                elif exercise_id == 4:  # 深蹲
                    down_threshold = max_angle - 2
                    up_threshold = min_angle + 2

                if self.debug and len(self.angle_history) == 50:
                    print(f"\n⚙️  [動態調整] 偵測到小幅度動作")
                    print(f"   角度範圍: {min_angle:.1f}° - {max_angle:.1f}° (變化 {angle_range:.1f}°)")
                    print(f"   調整閾值: DOWN={down_threshold:.1f}° | UP={up_threshold:.1f}°\n")

        # 統一的狀態機邏輯
        # 所有運動：DOWN階段 → UP階段 = 完成一次計數

        if exercise_id in [0, 1]:  # 二頭彎舉、錘式彎舉
            # 手臂伸直(大角度) = DOWN，彎曲(小角度) = UP
            if angle > down_threshold:
                self.stage = "down"
            elif angle < up_threshold:
                if self.stage == "down":
                    self.count += 1
                    if self.debug:
                        print(f"✓ [計數+1] 彎舉完成！角度: {angle:.1f}° | 總計: {self.count}")
                self.stage = "up"

        elif exercise_id == 2:  # 伏地挺身
            # 手臂彎曲(小角度) = DOWN，伸直(大角度) = UP
            if angle < down_threshold:
                self.stage = "down"
            elif angle > up_threshold:
                if self.stage == "down":
                    self.count += 1
                    if self.debug:
                        print(f"✓ [計數+1] 伏地挺身完成！角度: {angle:.1f}° | 總計: {self.count}")
                self.stage = "up"

        elif exercise_id == 3:  # 肩上推舉
            # 手臂收回(小角度) = DOWN，伸直向上(大角度) = UP
            if angle < down_threshold:
                self.stage = "down"
            elif angle > up_threshold:
                if self.stage == "down":
                    self.count += 1
                    if self.debug:
                        print(f"✓ [計數+1] 肩推完成！角度: {angle:.1f}° | 總計: {self.count}")
                self.stage = "up"

        elif exercise_id == 4:  # 深蹲
            # 蹲下(小角度) = DOWN，站立(大角度) = UP
            if angle < up_threshold:
                self.stage = "down"
            elif angle > down_threshold:
                if self.stage == "down":
                    self.count += 1
                    if self.debug:
                        print(f"✓ [計數+1] 深蹲完成！角度: {angle:.1f}° | 總計: {self.count}")
                self.stage = "up"

        if self.debug:
            print(f"[角度追蹤] 運動: {self.exercise_names[exercise_id]} | 角度: {angle:.1f}° | 階段: {self.stage} | 計數: {self.count}")

        result["count"] = self.count
        result["stage"] = self.stage

        return result

    def reset(self):
        """重置計數器"""
        self.count = 0
        self.stage = None
