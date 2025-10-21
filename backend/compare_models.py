#!/usr/bin/env python3
"""
比較兩個模型的差異
"""
import torch
import numpy as np

from model import BiLSTMAttention, BiLSTMSingleLayer
from feature_utils_v2 import landmarks_to_features_v2, landmarks_to_features_simple

print("=" * 70)
print("模型架構比較")
print("=" * 70)

# 原本的模型
model1 = BiLSTMAttention(input_dim=102, hidden_dim=96, attn_dim=128, num_classes=5)
checkpoint1 = torch.load("bilstm_mix_best_pt.pth", map_location="cpu", weights_only=True)
model1.load_state_dict(checkpoint1)
model1.eval()

# 新模型
model2 = BiLSTMSingleLayer(input_dim=22, hidden_dim=64, num_classes=5)
checkpoint2 = torch.load("model_bilstm.pth", map_location="cpu", weights_only=True)
model2.load_state_dict(checkpoint2)
model2.eval()

print("\n【模型 1: bilstm_mix_best_pt.pth】")
print("-" * 70)
print("✓ 架構:")
print("  - 雙層 BiLSTM (96 hidden units each)")
print("  - LayerNorm 正規化層 x2")
print("  - Attention 機制")
print("  - 全連接層 x2 (192→128→5)")
print(f"✓ 輸入特徵維度: 102 維")
print(f"✓ 參數數量: {sum(p.numel() for p in model1.parameters()):,}")
print(f"✓ 特徵提取: landmarks_to_features_v2 (角度 + 座標 + 幾何特徵)")

print("\n【模型 2: model_bilstm.pth】")
print("-" * 70)
print("✓ 架構:")
print("  - 單層 BiLSTM (64 hidden units)")
print("  - 全連接層 x1 (128→5)")
print(f"✓ 輸入特徵維度: 22 維")
print(f"✓ 參數數量: {sum(p.numel() for p in model2.parameters()):,}")
print(f"✓ 特徵提取: landmarks_to_features_simple (關節角度 + 基礎幾何)")

print("\n" + "=" * 70)
print("關鍵差異分析")
print("=" * 70)

print("\n【1. 模型容量】")
print(f"  原模型參數: 427,270")
print(f"  新模型參數:  45,701")
print(f"  → 原模型是新模型的 {427270/45701:.1f} 倍")
print(f"  → 更多參數 = 更強的表達能力")

print("\n【2. 特徵維度】")
print(f"  原模型: 102 維 (包含角度、座標、距離、速度等)")
print(f"  新模型:  22 維 (只有角度 + 基本幾何)")
print(f"  → 原模型有 {102/22:.1f} 倍的輸入資訊")

print("\n【3. 架構深度】")
print("  原模型: 雙層 LSTM + Attention + 雙層 FC")
print("  新模型: 單層 LSTM + 單層 FC")
print("  → 原模型層數更深，可以學習更複雜的特徵")

print("\n【4. 正規化技術】")
print("  原模型: LayerNorm (有助於穩定訓練)")
print("  新模型: 無")
print("  → LayerNorm 可以改善梯度流動，提升訓練效果")

print("\n【5. Attention 機制】")
print("  原模型: 有 (可以自動關注重要的時間步)")
print("  新模型: 無 (只取最後一個時間步)")
print("  → Attention 讓模型能識別動作的關鍵時刻")

print("\n" + "=" * 70)
print("為什麼原模型信心度高？")
print("=" * 70)

print("""
1. 【更豐富的特徵】
   - 102 維特徵包含更多資訊（角度、座標、速度等）
   - 可以捕捉更細微的動作差異

2. 【更深的網絡架構】
   - 雙層 LSTM 可以學習更抽象的時序模式
   - Attention 機制能聚焦在動作的關鍵幀

3. 【更多的參數】
   - 42.7 萬參數 vs 4.5 萬參數
   - 更強的模型容量 → 可以學習更複雜的決策邊界

4. 【正規化技術】
   - LayerNorm 穩定訓練過程
   - 避免梯度消失/爆炸

5. 【訓練品質】
   - 原模型可能經過更多 epoch 的訓練
   - 使用更好的超參數和訓練策略
""")

print("\n" + "=" * 70)
print("實際輸出比較（隨機輸入測試）")
print("=" * 70)

# 生成假的 landmarks 來測試
fake_landmarks = [np.random.randn(33, 4).astype(np.float32) for _ in range(45)]

# 提取特徵
features_102 = landmarks_to_features_v2(fake_landmarks)
features_22 = landmarks_to_features_simple(fake_landmarks)

print(f"\n特徵形狀:")
print(f"  102 維: {features_102.shape}")
print(f"   22 維: {features_22.shape}")

# 推論
with torch.no_grad():
    # 原模型
    x1 = torch.FloatTensor([features_102])
    out1 = model1(x1)
    prob1 = torch.softmax(out1, dim=1)[0]

    # 新模型
    x2 = torch.FloatTensor([features_22])
    out2 = model2(x2)
    prob2 = torch.softmax(out2, dim=1)[0]

exercise_names = [
    "Barbell Biceps Curl",
    "Hammer Curl",
    "Push-up",
    "Shoulder Press",
    "Squat"
]

print(f"\n原模型輸出機率:")
for i, (name, p) in enumerate(zip(exercise_names, prob1)):
    print(f"  {name:20s}: {p:.4f}")
print(f"  最大信心度: {prob1.max():.4f}")
print(f"  機率標準差: {prob1.std():.4f}")

print(f"\n新模型輸出機率:")
for i, (name, p) in enumerate(zip(exercise_names, prob2)):
    print(f"  {name:20s}: {p:.4f}")
print(f"  最大信心度: {prob2.max():.4f}")
print(f"  機率標準差: {prob2.std():.4f}")

print("\n→ 信心度高 = 機率分布更集中（標準差大）")
print("→ 信心度低 = 機率分布平均（標準差小）")
