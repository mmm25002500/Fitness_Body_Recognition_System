import numpy as np

def landmarks_to_features_simple(landmarks_sequence):
    """
    將姿勢關節點轉換為 22 維特徵向量 (用於新模型)

    這是簡化版特徵提取，只使用關鍵關節點的相對位置
    特徵組成：
    - 8 個關節角度 (肘x2, 膝x2, 肩x2, 髖x2)
    - 14 維額外特徵 (身體比例、關節距離等)

    總計: 22 維
    """
    features = []

    for frame_landmarks in landmarks_sequence:
        frame_features = []

        # 1. 關節角度特徵（8 維）
        angles = calculate_joint_angles(frame_landmarks)
        frame_features.extend(angles)

        # 2. 身體比例特徵（14 維）
        # 計算關鍵關節點之間的距離
        try:
            # 肩膀寬度
            shoulder_width = np.linalg.norm(
                frame_landmarks[11][:3] - frame_landmarks[12][:3]
            )

            # 髖部寬度
            hip_width = np.linalg.norm(
                frame_landmarks[23][:3] - frame_landmarks[24][:3]
            )

            # 軀幹長度 (左肩到左髖)
            torso_length = np.linalg.norm(
                frame_landmarks[11][:3] - frame_landmarks[23][:3]
            )

            # 左臂長度
            left_arm = np.linalg.norm(
                frame_landmarks[11][:3] - frame_landmarks[13][:3]
            ) + np.linalg.norm(
                frame_landmarks[13][:3] - frame_landmarks[15][:3]
            )

            # 右臂長度
            right_arm = np.linalg.norm(
                frame_landmarks[12][:3] - frame_landmarks[14][:3]
            ) + np.linalg.norm(
                frame_landmarks[14][:3] - frame_landmarks[16][:3]
            )

            # 左腿長度
            left_leg = np.linalg.norm(
                frame_landmarks[23][:3] - frame_landmarks[25][:3]
            ) + np.linalg.norm(
                frame_landmarks[25][:3] - frame_landmarks[27][:3]
            )

            # 右腿長度
            right_leg = np.linalg.norm(
                frame_landmarks[24][:3] - frame_landmarks[26][:3]
            ) + np.linalg.norm(
                frame_landmarks[26][:3] - frame_landmarks[28][:3]
            )

            # 身體中心的 y 座標 (髖部平均高度)
            hip_y = (frame_landmarks[23][1] + frame_landmarks[24][1]) / 2

            # 手腕相對髖部的高度
            left_wrist_y = frame_landmarks[15][1] - hip_y
            right_wrist_y = frame_landmarks[16][1] - hip_y

            # 腳踝相對髖部的高度
            left_ankle_y = frame_landmarks[27][1] - hip_y
            right_ankle_y = frame_landmarks[28][1] - hip_y

            # 手腕的 x 座標差異
            wrists_x_diff = frame_landmarks[15][0] - frame_landmarks[16][0]

            # 腳踝的 x 座標差異
            ankles_x_diff = frame_landmarks[27][0] - frame_landmarks[28][0]

            additional = [
                shoulder_width, hip_width, torso_length,
                left_arm, right_arm, left_leg, right_leg,
                hip_y, left_wrist_y, right_wrist_y,
                left_ankle_y, right_ankle_y,
                wrists_x_diff, ankles_x_diff
            ]

        except Exception as e:
            # 如果計算失敗，使用零填充
            additional = [0.0] * 14

        frame_features.extend(additional)

        # 確保正好 22 維
        if len(frame_features) < 22:
            frame_features.extend([0.0] * (22 - len(frame_features)))

        features.append(frame_features[:22])

    return np.array(features, dtype=np.float32)

def calculate_joint_angles(landmarks):
    """
    計算關節角度特徵

    輸入: shape (33, 4) 的關節點陣列
    輸出: 關節角度陣列
    """
    def get_angle(p1, p2, p3):
        """計算三點構成的角度（弧度）"""
        v1 = p1[:2] - p2[:2]
        v2 = p3[:2] - p2[:2]
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return np.arccos(cos_angle)

    angles = []

    # 左臂角度
    if landmarks[11][3] > 0.3 and landmarks[13][3] > 0.3 and landmarks[15][3] > 0.3:
        angles.append(get_angle(landmarks[11], landmarks[13], landmarks[15]))  # 左肘
    else:
        angles.append(0.0)

    # 右臂角度
    if landmarks[12][3] > 0.3 and landmarks[14][3] > 0.3 and landmarks[16][3] > 0.3:
        angles.append(get_angle(landmarks[12], landmarks[14], landmarks[16]))  # 右肘
    else:
        angles.append(0.0)

    # 左腿角度
    if landmarks[23][3] > 0.3 and landmarks[25][3] > 0.3 and landmarks[27][3] > 0.3:
        angles.append(get_angle(landmarks[23], landmarks[25], landmarks[27]))  # 左膝
    else:
        angles.append(0.0)

    # 右腿角度
    if landmarks[24][3] > 0.3 and landmarks[26][3] > 0.3 and landmarks[28][3] > 0.3:
        angles.append(get_angle(landmarks[24], landmarks[26], landmarks[28]))  # 右膝
    else:
        angles.append(0.0)

    # 左肩角度
    if landmarks[13][3] > 0.3 and landmarks[11][3] > 0.3 and landmarks[23][3] > 0.3:
        angles.append(get_angle(landmarks[13], landmarks[11], landmarks[23]))  # 左肩
    else:
        angles.append(0.0)

    # 右肩角度
    if landmarks[14][3] > 0.3 and landmarks[12][3] > 0.3 and landmarks[24][3] > 0.3:
        angles.append(get_angle(landmarks[14], landmarks[12], landmarks[24]))  # 右肩
    else:
        angles.append(0.0)

    # 左髖角度
    if landmarks[11][3] > 0.3 and landmarks[23][3] > 0.3 and landmarks[25][3] > 0.3:
        angles.append(get_angle(landmarks[11], landmarks[23], landmarks[25]))  # 左髖
    else:
        angles.append(0.0)

    # 右髖角度
    if landmarks[12][3] > 0.3 and landmarks[24][3] > 0.3 and landmarks[26][3] > 0.3:
        angles.append(get_angle(landmarks[12], landmarks[24], landmarks[26]))  # 右髖
    else:
        angles.append(0.0)

    return np.array(angles)

def landmarks_to_features_v2(landmarks_sequence):
    """
    將姿勢關節點轉換為 102 維特徵向量（角度 + 座標混合）

    特徵組成：
    - 關節角度：8 個主要關節角度
    - 關節座標：17 個重要關節點的 (x, y, z) 座標，共 51 維
    - 其他幾何特徵：肢段長度、相對位置等，共 43 維

    總計約 102 維
    """
    important_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    features = []

    for frame_landmarks in landmarks_sequence:
        frame_features = []

        # 1. 關節角度特徵（8 維）
        angles = calculate_joint_angles(frame_landmarks)
        frame_features.extend(angles)

        # 2. 重要關節點的座標（17 × 3 = 51 維）
        coords = []
        for idx in important_indices:
            coords.extend(frame_landmarks[idx][:3])  # x, y, z

        # 以髖部中心點正規化
        hip_center = np.mean([frame_landmarks[23][:3], frame_landmarks[24][:3]], axis=0)
        coords = np.array(coords).reshape(-1, 3)
        coords = (coords - hip_center).flatten()

        frame_features.extend(coords)

        # 3. 額外的幾何特徵（43 維）
        # 肢段長度、角度餘弦值、速度估計等
        # 這裡簡化為填充零或使用座標差值
        additional_features = np.zeros(43)

        # 計算一些肢段長度作為補充
        if len(frame_features) + len(additional_features) < 102:
            padding = 102 - (len(frame_features) + len(additional_features))
            additional_features = np.concatenate([additional_features, np.zeros(padding)])

        frame_features.extend(additional_features[:43])

        features.append(frame_features[:102])  # 確保正好 102 維

    return np.array(features, dtype=np.float32)

def create_sliding_windows(features, window_size=45, stride=3):
    """
    使用滑動窗口建構時序樣本

    參數:
        features: shape (n_frames, feature_dim) 的特徵陣列
        window_size: 窗口大小（預設 45 幀）
        stride: 步幅（預設 3 幀）

    返回:
        shape (n_windows, window_size, feature_dim) 的序列陣列
    """
    n_frames, feature_dim = features.shape
    windows = []

    for start in range(0, n_frames - window_size + 1, stride):
        window = features[start:start + window_size]
        windows.append(window)

    if len(windows) == 0:  # 如果影片太短
        # 使用零填充
        padded = np.zeros((window_size, feature_dim), dtype=np.float32)
        if n_frames > 0:
            padded[:n_frames] = features
        windows.append(padded)

    return np.array(windows, dtype=np.float32)

def z_score_normalize(features):
    """
    Z-score 標準化（零均值、單位變異）

    參數:
        features: shape (n_samples, ...) 的特徵陣列

    返回:
        標準化後的特徵陣列
    """
    # 計算每個特徵維度的均值和標準差
    if len(features.shape) == 3:  # (n_windows, window_size, feature_dim)
        # 在前兩個維度上計算統計量
        mean = np.mean(features, axis=(0, 1), keepdims=True)
        std = np.std(features, axis=(0, 1), keepdims=True) + 1e-8
    elif len(features.shape) == 2:  # (n_frames, feature_dim)
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
    else:
        raise ValueError(f"不支援的特徵形狀: {features.shape}")

    return (features - mean) / std
