#!/usr/bin/env python3
"""
資料預處理腳本
從影片提取姿勢特徵，準備訓練資料
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from pose_extractor_mediapipe import PoseExtractorMediaPipe
from feature_utils_v2 import landmarks_to_features_v2, create_sliding_windows
from training_config import (
    DATASET_PATH, CLASS_TO_ID, FEATURE_CONFIG,
    OUTPUT_PATHS, TRAINING_CONFIG
)


def process_video(video_path, pose_extractor):
    """
    處理單個影片，提取姿勢特徵

    Returns:
        features: numpy array of shape (n_frames, feature_dim)
        或 None 如果處理失敗
    """
    cap = cv2.VideoCapture(video_path)
    frame_landmarks = []

    frame_count = 0
    success_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 提取姿勢關節點
        landmarks, _ = pose_extractor.extract_landmarks(frame)

        if landmarks is not None:
            frame_landmarks.append(landmarks)
            success_count += 1

    cap.release()

    # 檢查是否有足夠的幀
    if len(frame_landmarks) < FEATURE_CONFIG['min_frames']:
        print(f"  ⚠️  影片太短: {video_path.name} ({len(frame_landmarks)} 幀 < {FEATURE_CONFIG['min_frames']})")
        return None

    # 轉換為特徵向量
    features = landmarks_to_features_v2(frame_landmarks)

    print(f"  ✓ {video_path.name}: {frame_count} 幀, {success_count} 有效, {len(features)} 特徵")

    return features


def create_windows_from_features(features, class_id):
    """
    從特徵序列建立滑動窗口

    Returns:
        windows: list of (window_features, class_id)
    """
    windows = create_sliding_windows(
        features,
        window_size=FEATURE_CONFIG['window_size'],
        stride=FEATURE_CONFIG['stride']
    )

    # 每個窗口配對類別標籤
    return [(window, class_id) for window in windows]


def process_dataset():
    """
    處理整個資料集
    """
    print("=" * 60)
    print("開始資料預處理")
    print("=" * 60)

    # 初始化 MediaPipe
    print("\n初始化 MediaPipe...")
    pose_extractor = PoseExtractorMediaPipe()
    print("✓ MediaPipe 初始化完成")

    # 準備輸出目錄
    output_dir = Path(OUTPUT_PATHS['processed_data'])
    output_dir.mkdir(exist_ok=True, parents=True)

    # 收集所有影片
    all_videos = []
    dataset_path = Path(DATASET_PATH)

    print(f"\n掃描資料集: {dataset_path}")
    for class_name, class_id in CLASS_TO_ID.items():
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"⚠️  找不到類別資料夾: {class_dir}")
            continue

        videos = list(class_dir.glob("*.mp4"))
        print(f"  {class_name}: {len(videos)} 個影片")

        for video_path in videos:
            all_videos.append((video_path, class_id, class_name))

    print(f"\n總共找到 {len(all_videos)} 個影片")

    # 處理所有影片
    all_windows = []
    class_stats = {i: 0 for i in range(len(CLASS_TO_ID))}

    print("\n開始處理影片...")
    for video_path, class_id, class_name in tqdm(all_videos, desc="處理影片"):
        features = process_video(video_path, pose_extractor)

        if features is None:
            continue

        # 建立滑動窗口
        windows = create_windows_from_features(features, class_id)
        all_windows.extend(windows)
        class_stats[class_id] += len(windows)

    print(f"\n✓ 成功處理 {len(all_videos)} 個影片")
    print(f"✓ 生成 {len(all_windows)} 個訓練樣本")

    # 顯示每個類別的樣本數
    print("\n各類別樣本數:")
    for class_name, class_id in CLASS_TO_ID.items():
        print(f"  {class_name}: {class_stats[class_id]} 個窗口")

    # 轉換為 numpy 陣列
    print("\n準備訓練資料...")
    X = np.array([window for window, _ in all_windows], dtype=np.float32)
    y = np.array([label for _, label in all_windows], dtype=np.int64)

    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")

    # 資料分割
    from sklearn.model_selection import train_test_split

    val_split = TRAINING_CONFIG['val_split']
    test_split = TRAINING_CONFIG['test_split']
    train_split = 1 - val_split - test_split

    print(f"\n資料分割: 訓練 {train_split:.0%} / 驗證 {val_split:.0%} / 測試 {test_split:.0%}")

    # 先分出測試集
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42, stratify=y
    )

    # 再分出驗證集
    val_size_adjusted = val_split / (1 - test_split)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
    )

    print(f"  訓練集: {X_train.shape[0]} 樣本")
    print(f"  驗證集: {X_val.shape[0]} 樣本")
    print(f"  測試集: {X_test.shape[0]} 樣本")

    # 儲存資料
    print("\n儲存資料...")
    np.savez_compressed(
        output_dir / 'train.npz',
        X=X_train, y=y_train
    )
    print(f"  ✓ {output_dir / 'train.npz'}")

    np.savez_compressed(
        output_dir / 'val.npz',
        X=X_val, y=y_val
    )
    print(f"  ✓ {output_dir / 'val.npz'}")

    np.savez_compressed(
        output_dir / 'test.npz',
        X=X_test, y=y_test
    )
    print(f"  ✓ {output_dir / 'test.npz'}")

    # 儲存資料集資訊
    dataset_info = {
        'num_videos': len(all_videos),
        'num_samples': len(all_windows),
        'num_classes': len(CLASS_TO_ID),
        'class_names': list(CLASS_TO_ID.keys()),
        'class_distribution': class_stats,
        'train_size': int(X_train.shape[0]),
        'val_size': int(X_val.shape[0]),
        'test_size': int(X_test.shape[0]),
        'feature_dim': int(X.shape[2]),
        'window_size': int(X.shape[1]),
    }

    with open(output_dir / 'dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)
    print(f"  ✓ {output_dir / 'dataset_info.json'}")

    print("\n" + "=" * 60)
    print("✅ 資料預處理完成！")
    print("=" * 60)
    print(f"\n輸出位置: {output_dir}")
    print("\n下一步: 執行 train_exercise_model.py 開始訓練")


if __name__ == "__main__":
    # 設置環境變數以減少 TensorFlow 日誌
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        process_dataset()
    except KeyboardInterrupt:
        print("\n\n⚠️  使用者中斷")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
