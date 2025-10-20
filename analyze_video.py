"""
影片動作片段分析工具
檢測影片中包含哪些動作及其時間分佈
"""

import torch
import numpy as np
from inference_v2 import ExerciseClassifierV2
from collections import Counter
import sys

def analyze_video_segments(video_path, window_size=45, stride=3):
    """分析影片中的動作片段"""

    classifier = ExerciseClassifierV2()

    print(f"正在分析影片: {video_path}\n")

    # 獲取所有窗口的預測
    all_logits, all_probs = classifier.predict_video_sequence_level(
        video_path, window_size=window_size, stride=stride
    )

    # 每個窗口的預測
    window_predictions = np.argmax(all_probs, axis=1)
    window_confidences = np.max(all_probs, axis=1)

    # 統計整體分佈
    print("\n=== 動作分佈統計 ===")
    counter = Counter(window_predictions)
    total_windows = len(window_predictions)

    for class_idx, count in counter.most_common():
        percentage = count / total_windows * 100
        print(f"{classifier.class_names[class_idx]}: {count} 個窗口 ({percentage:.1f}%)")

    # 找出連續片段
    print("\n=== 連續動作片段 ===")
    segments = []
    current_action = window_predictions[0]
    start_window = 0

    for i in range(1, len(window_predictions)):
        if window_predictions[i] != current_action:
            # 新片段開始
            avg_confidence = np.mean(window_confidences[start_window:i])
            segments.append({
                'action': current_action,
                'start': start_window,
                'end': i - 1,
                'confidence': avg_confidence
            })
            current_action = window_predictions[i]
            start_window = i

    # 最後一個片段
    avg_confidence = np.mean(window_confidences[start_window:])
    segments.append({
        'action': current_action,
        'start': start_window,
        'end': len(window_predictions) - 1,
        'confidence': avg_confidence
    })

    # 顯示片段（只顯示長度 >= 3 的片段，過濾雜訊）
    print("\n片段編號 | 動作 | 窗口範圍 | 平均信心度")
    print("-" * 70)

    segment_num = 1
    for seg in segments:
        segment_length = seg['end'] - seg['start'] + 1
        if segment_length >= 3:  # 只顯示較長片段
            action_name = classifier.class_names[seg['action']]
            print(f"片段 {segment_num:2d}   | {action_name:30s} | {seg['start']:3d}-{seg['end']:3d} ({segment_length:2d}個) | {seg['confidence']:.1%}")
            segment_num += 1

    # 估算時間（假設 30 FPS）
    fps = 30
    frame_per_window = stride

    print(f"\n=== 時間軸估計（假設 {fps} FPS）===")
    for i, seg in enumerate(segments):
        if seg['end'] - seg['start'] + 1 >= 3:
            start_frame = seg['start'] * frame_per_window
            end_frame = seg['end'] * frame_per_window + window_size
            start_time = start_frame / fps
            end_time = end_frame / fps

            action_name = classifier.class_names[seg['action']]
            print(f"{action_name}: {start_time:.1f}s - {end_time:.1f}s")

    # 推薦主要動作
    print("\n=== 主要動作判定 ===")

    # 方法1: 數量最多
    most_common_action = counter.most_common(1)[0][0]
    print(f"方法1（窗口數最多）: {classifier.class_names[most_common_action]}")

    # 方法2: 信心度最高的片段
    high_conf_segments = [s for s in segments if s['end'] - s['start'] >= 5 and s['confidence'] > 0.9]
    if high_conf_segments:
        best_segment = max(high_conf_segments, key=lambda x: x['confidence'])
        print(f"方法2（高信心度片段）: {classifier.class_names[best_segment['action']]} (信心度: {best_segment['confidence']:.1%})")

    # 方法3: 中間片段
    middle_window = len(window_predictions) // 2
    middle_action = window_predictions[middle_window]
    print(f"方法3（影片中段動作）: {classifier.class_names[middle_action]}")

    return segments, all_probs

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方式: python analyze_video.py <影片路徑>")
        print("範例: python analyze_video.py example_video.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    analyze_video_segments(video_path)
