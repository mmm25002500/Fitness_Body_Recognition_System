#!/usr/bin/env python3
"""
測試 BiLSTM 模型的預測功能
使用方式: python test_model.py <video_path>
"""
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

from pose_extractor_mediapipe import PoseExtractorMediaPipe
from model import BiLSTMAttention
from feature_utils_v2 import landmarks_to_features_v2

# 運動類別
exercise_names = [
    "Barbell Biceps Curl",
    "Hammer Curl",
    "Push-up",
    "Shoulder Press",
    "Squat"
]

def test_video(video_path, window_size=45, stride=3):
    """測試單一影片 - 使用滑動窗口預測"""
    print(f"🧪 測試影片: {video_path}")
    print("=" * 60)

    # 載入模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"📱 使用設備: {device}")

    model = BiLSTMAttention(
        input_dim=102,
        hidden_dim=96,
        attn_dim=128,
        num_classes=5
    )
    checkpoint = torch.load("bilstm_mix_best_pt.pth", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    print("✓ 模型載入成功 (BiLSTMAttention, 102 dims)")

    # 初始化姿態提取器
    pose_extractor = PoseExtractorMediaPipe()
    print("✓ MediaPipe 初始化完成")

    # 讀取影片
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 無法開啟影片: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📹 影片資訊: {frame_count} 幀, FPS={fps:.1f}")
    print()

    # 提取特徵
    frame_features = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        landmarks, _ = pose_extractor.extract_landmarks(frame)
        if landmarks is not None:
            features = landmarks_to_features_v2([landmarks])[0]
            frame_features.append(features)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"處理中... {frame_idx}/{frame_count} 幀")

    cap.release()

    print(f"\n✓ 成功提取 {len(frame_features)} 幀特徵")
    print(f"✓ 特徵維度: {len(frame_features[0]) if frame_features else 0}")

    if len(frame_features) < window_size:
        print(f"❌ 影片太短！至少需要 {window_size} 幀，但只有 {len(frame_features)} 幀")
        return None

    # 滑動窗口預測
    predictions = []
    prediction_details = []

    print(f"\n🔄 使用滑動窗口預測 (window_size={window_size}, stride={stride})...")

    for i in range(0, len(frame_features) - window_size + 1, stride):
        window = frame_features[i:i + window_size]
        window_tensor = torch.FloatTensor([window]).to(device)

        with torch.no_grad():
            output = model(window_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][pred_class].item()

            predictions.append(pred_class)
            prediction_details.append({
                "window": i,
                "class": pred_class,
                "confidence": confidence,
                "probabilities": probabilities[0].cpu().numpy()
            })

    # 統計結果
    class_counts = {}
    for pred in predictions:
        class_counts[pred] = class_counts.get(pred, 0) + 1

    print(f"\n📊 預測結果統計 (共 {len(predictions)} 個窗口):")
    print("-" * 60)
    for class_id in range(5):
        count = class_counts.get(class_id, 0)
        if count > 0:
            percentage = count / len(predictions) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"{exercise_names[class_id]:20s}: {bar:50s} {count:3d} ({percentage:5.1f}%)")

    # 最終預測
    final_class = max(class_counts, key=class_counts.get)
    avg_confidence = np.mean([
        d["confidence"] for d in prediction_details if d["class"] == final_class
    ])

    print(f"\n🎯 最終預測: {exercise_names[final_class]}")
    print(f"   平均信心度: {avg_confidence:.2%}")

    # 顯示詳細機率分布
    print(f"\n📈 各類別平均機率:")
    print("-" * 60)
    all_probs = np.array([d["probabilities"] for d in prediction_details])
    avg_probs = np.mean(all_probs, axis=0)

    for i, (name, prob) in enumerate(zip(exercise_names, avg_probs)):
        bar_length = int(prob * 50)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        marker = " ← 最終預測" if i == final_class else ""
        print(f"{name:20s}: {bar} {prob:.2%}{marker}")

    return {
        "final_class": final_class,
        "final_name": exercise_names[final_class],
        "confidence": avg_confidence,
        "class_counts": class_counts,
        "total_predictions": len(predictions),
        "avg_probs": avg_probs
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式: python test_model.py <video_path>")
        print("範例: python test_model.py ../example_video1.mp4")
        sys.exit(1)

    video_path = sys.argv[1]

    if not Path(video_path).exists():
        print(f"❌ 找不到影片檔案: {video_path}")
        sys.exit(1)

    result = test_video(video_path)

    if result:
        print("\n" + "=" * 60)
        print("✅ 測試完成！")
        print("=" * 60)

        if result['confidence'] > 0.5:
            print(f"✅ 高信心度預測: {result['final_name']} ({result['confidence']:.2%})")
        else:
            print(f"⚠️  低信心度預測: {result['final_name']} ({result['confidence']:.2%})")
            print("   建議檢查影片品質或訓練資料")
