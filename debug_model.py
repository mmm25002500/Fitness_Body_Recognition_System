"""
模型診斷工具：檢查模型輸入輸出和特徵
"""

import torch
import numpy as np
from test import BiLSTMAttention
from pose_extractor import PoseExtractor
from feature_utils_v2 import landmarks_to_features_v2, create_sliding_windows, z_score_normalize

def debug_model_input():
    """檢查模型輸入格式"""
    print("=== 模型架構檢查 ===\n")

    model = BiLSTMAttention(input_dim=102, hidden_dim=96, attn_dim=128, num_classes=5)
    state_dict = torch.load("bilstm_mix_best_pt.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("模型載入成功！")
    print(f"輸入維度: 102")
    print(f"隱藏層維度: 96")
    print(f"注意力維度: 128")
    print(f"類別數: 5")

    # 測試隨機輸入
    print("\n=== 測試隨機輸入 ===")
    test_input = torch.randn(1, 45, 102)
    with torch.no_grad():
        output = model(test_input)
        probs = torch.softmax(output, dim=1)

    print(f"輸入形狀: {test_input.shape}")
    print(f"輸出形狀: {output.shape}")
    print(f"輸出 logits: {output.numpy()}")
    print(f"輸出機率: {probs.numpy()}")

def debug_feature_extraction(video_path):
    """檢查特徵提取過程"""
    print(f"\n=== 特徵提取診斷 ===\n")
    print(f"影片: {video_path}")

    # 1. 提取姿勢
    extractor = PoseExtractor()
    landmarks_sequence = extractor.process_video(video_path, max_frames=50)

    if landmarks_sequence is None:
        print("錯誤: 無法提取姿勢")
        return

    print(f"姿勢序列形狀: {landmarks_sequence.shape}")
    print(f"範例關節點 (第1幀, 前3個點):\n{landmarks_sequence[0][:3]}")

    # 2. 轉換為特徵
    features = landmarks_to_features_v2(landmarks_sequence)
    print(f"\n特徵形狀: {features.shape}")
    print(f"特徵範圍: min={features.min():.3f}, max={features.max():.3f}, mean={features.mean():.3f}")

    # 3. 標準化
    features_normalized = z_score_normalize(features)
    print(f"\n標準化後:")
    print(f"特徵範圍: min={features_normalized.min():.3f}, max={features_normalized.max():.3f}, mean={features_normalized.mean():.3f}")

    # 4. 建立窗口
    windows = create_sliding_windows(features_normalized, window_size=45, stride=3)
    print(f"\n滑動窗口數量: {len(windows)}")
    print(f"窗口形狀: {windows.shape}")

    # 5. 測試第一個窗口的預測
    model = BiLSTMAttention(input_dim=102, hidden_dim=96, attn_dim=128, num_classes=5)
    state_dict = torch.load("bilstm_mix_best_pt.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("\n=== 單一窗口預測 ===")
    with torch.no_grad():
        x = torch.FloatTensor(windows[0]).unsqueeze(0)
        output = model(x)
        probs = torch.softmax(output, dim=1).numpy()[0]

    class_names = [
        "槓鈴二頭彎舉",
        "錘式彎舉",
        "伏地挺身",
        "肩上推舉",
        "深蹲"
    ]

    print("第一個窗口的預測機率:")
    for name, prob in zip(class_names, probs):
        print(f"  {name}: {prob:.2%}")

    # 6. 檢查所有窗口的預測分佈
    print("\n=== 所有窗口預測分佈 ===")
    all_predictions = []
    with torch.no_grad():
        for window in windows[:20]:  # 只檢查前20個
            x = torch.FloatTensor(window).unsqueeze(0)
            output = model(x)
            pred_class = torch.argmax(output, dim=1).item()
            all_predictions.append(pred_class)

    print("前20個窗口的預測:")
    for i, pred in enumerate(all_predictions):
        print(f"  窗口 {i}: {class_names[pred]}")

    # 統計
    from collections import Counter
    counter = Counter(all_predictions)
    print("\n預測統計:")
    for class_idx, count in counter.items():
        print(f"  {class_names[class_idx]}: {count} 次")

if __name__ == "__main__":
    import sys

    print("模型診斷工具\n")

    # 檢查模型
    debug_model_input()

    # 檢查特徵提取
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        debug_feature_extraction(video_path)
    else:
        print("\n提示: 使用 python debug_model.py <影片路徑> 來診斷特定影片")
