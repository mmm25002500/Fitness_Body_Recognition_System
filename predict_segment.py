"""
預測影片特定時間段的動作
"""

import sys
from inference_v2 import ExerciseClassifierV2
from pose_extractor import PoseExtractor
from feature_utils_v2 import landmarks_to_features_v2, create_sliding_windows, z_score_normalize
import torch
import numpy as np
import cv2

def predict_video_segment(video_path, start_sec=None, end_sec=None):
    """
    預測影片特定時間段的動作

    參數:
        video_path: 影片路徑
        start_sec: 起始時間（秒）
        end_sec: 結束時間（秒）
    """
    print(f"正在分析影片: {video_path}")

    # 獲取影片資訊
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"影片資訊: {fps:.1f} FPS, {total_frames} 幀, 總時長 {duration:.1f} 秒")

    # 計算幀範圍
    start_frame = int(start_sec * fps) if start_sec else 0
    end_frame = int(end_sec * fps) if end_sec else total_frames

    print(f"分析範圍: {start_frame} - {end_frame} 幀 ({start_frame/fps:.1f}s - {end_frame/fps:.1f}s)")

    # 提取該時段的姿勢
    extractor = PoseExtractor()
    cap = cv2.VideoCapture(video_path)

    all_landmarks = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= end_frame:
            break

        if frame_idx >= start_frame:
            landmarks, _ = extractor.extract_landmarks(frame)
            if landmarks is not None:
                all_landmarks.append(landmarks)

        frame_idx += 1

    cap.release()

    if len(all_landmarks) == 0:
        print("錯誤: 無法提取姿勢")
        return

    all_landmarks = np.array(all_landmarks)
    print(f"成功提取 {len(all_landmarks)} 幀姿勢")

    # 轉換特徵並預測
    features = landmarks_to_features_v2(all_landmarks)
    features = z_score_normalize(features)
    windows = create_sliding_windows(features, window_size=45, stride=3)

    print(f"建立 {len(windows)} 個滑動窗口")

    # 載入模型預測
    classifier = ExerciseClassifierV2()

    all_probs = []
    with torch.no_grad():
        for window in windows:
            x = torch.FloatTensor(window).unsqueeze(0).to(classifier.device)
            output = classifier.model(x)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    all_probs = np.array(all_probs)

    # 加權平均所有窗口
    window_confidences = np.max(all_probs, axis=1)
    weights = window_confidences / np.sum(window_confidences)
    final_probs = np.average(all_probs, axis=0, weights=weights)

    predicted_class = np.argmax(final_probs)
    confidence = final_probs[predicted_class]

    print(f"\n預測結果: {classifier.class_names[predicted_class]}")
    print(f"信心度: {confidence:.2%}")

    print("\n所有類別機率:")
    for i, (name, prob) in enumerate(zip(classifier.class_names, final_probs)):
        marker = " ← 預測" if i == predicted_class else ""
        print(f"  {name}: {prob:.2%}{marker}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式: python predict_segment.py <影片路徑> [起始秒數] [結束秒數]")
        print("範例:")
        print("  python predict_segment.py example_video.mp4          # 分析整個影片")
        print("  python predict_segment.py example_video.mp4 5 10     # 分析 5-10 秒")
        sys.exit(1)

    video_path = sys.argv[1]
    start_sec = float(sys.argv[2]) if len(sys.argv) > 2 else None
    end_sec = float(sys.argv[3]) if len(sys.argv) > 3 else None

    predict_video_segment(video_path, start_sec, end_sec)
