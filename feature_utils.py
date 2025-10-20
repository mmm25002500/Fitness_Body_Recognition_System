import numpy as np

def landmarks_to_features(landmarks_sequence):
    """
    將 MediaPipe 姿勢關節點轉換為 102 維特徵向量序列

    輸入: shape (n_frames, 33, 4) - 33 個關節點，每個有 [x, y, z, visibility]
    輸出: shape (n_frames, 102) - 每幀 102 維特徵

    特徵組成：
    - 選取重要的 17 個關節點 (上半身、下半身主要關節)
    - 每個關節點使用 (x, y, z) 座標，忽略 visibility
    - 進行正規化處理
    """
    # MediaPipe 重要關節點索引
    # 上半身：鼻子(0), 左右肩(11,12), 左右手肘(13,14), 左右手腕(15,16)
    # 核心：左右髖(23,24)
    # 下半身：左右膝蓋(25,26), 左右腳踝(27,28), 左右腳尖(31,32), 左右腳跟(29,30)
    important_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

    # 提取重要關節點的 x, y, z 座標
    features = []
    for frame_landmarks in landmarks_sequence:
        frame_features = []
        for idx in important_indices:
            # 只取 x, y, z，忽略 visibility
            frame_features.extend(frame_landmarks[idx][:3])

        # 正規化：以髖部中心點為原點
        hip_center = np.mean([frame_landmarks[23][:3], frame_landmarks[24][:3]], axis=0)
        frame_features = np.array(frame_features).reshape(-1, 3)
        frame_features = (frame_features - hip_center).flatten()

        features.append(frame_features)

    return np.array(features, dtype=np.float32)

def prepare_sequence(features, target_length=30):
    """
    準備固定長度的序列用於模型輸入

    - 若序列過短：使用零填充
    - 若序列過長：取最後 target_length 幀
    """
    n_frames, feature_dim = features.shape

    if n_frames >= target_length:
        return features[-target_length:]
    else:
        # 零填充
        padded = np.zeros((target_length, feature_dim), dtype=np.float32)
        padded[-n_frames:] = features
        return padded

def normalize_features(features):
    """對特徵進行標準化（zero mean, unit variance）"""
    mean = np.mean(features, axis=0, keepdims=True)
    std = np.std(features, axis=0, keepdims=True) + 1e-8
    return (features - mean) / std
