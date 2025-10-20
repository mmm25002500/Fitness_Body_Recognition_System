# 運動動作分類系統 - 環境設置指南

本專題使用深度學習模型（BiLSTM + Attention）對運動影片進行分類。為了達到最佳準確率，建議使用 **MediaPipe** 作為姿勢估計器。

---

## 📋 系統需求

- **推薦配置**：Python 3.11 + MediaPipe（準確率 85-90%）
- **替代配置**：Python 3.14 + YOLOv8（準確率 60-70%）

---

## 🎯 方案 1: 使用 MediaPipe（推薦）

### 優點
✅ 完整 33 個關節點（包含手部細節）
✅ 3D 座標 (x, y, z)
✅ 與訓練模型完全匹配
✅ 準確率最高（87%+）

### 安裝步驟

#### 步驟 1: 安裝 Python 3.11

**選項 A: 使用 pyenv（推薦）**
```bash
# 安裝 pyenv
brew install pyenv

# 安裝 Python 3.11.9
pyenv install 3.11.9

# 設定專案使用的 Python 版本
cd /Users/tershi/Project/專題
pyenv local 3.11.9

# 驗證版本
python --version  # 應該顯示 Python 3.11.9
```

**選項 B: 使用 Conda**
```bash
# 創建新環境
conda create -n exercise-classifier python=3.11

# 啟用環境
conda activate exercise-classifier
```

#### 步驟 2: 安裝套件
```bash
# 安裝相依套件
pip install torch>=2.0.0
pip install opencv-python>=4.8.0
pip install mediapipe==0.10.14
pip install numpy>=1.24.0
```

#### 步驟 3: 驗證安裝
```bash
# 檢查系統配置
python pose_config.py
```

應該顯示：
```
=== 系統配置 ===
Python 版本: 3.11.9
MediaPipe: ✓ 0.10.14
YOLOv8: ✗ 未安裝

推薦: 使用 MediaPipe 以獲得最佳準確率
```

#### 步驟 4: 測試姿勢提取
```bash
# 測試 MediaPipe 提取器
python pose_extractor_mediapipe.py example_video.mp4
```

---

## 🔄 方案 2: 使用 YOLOv8（Python 3.14）

### 優點
✅ 支援最新 Python 版本
✅ 速度較快
⚠️  準確率較低（60-70%）

### 安裝步驟

```bash
# 直接安裝套件（當前環境已是 Python 3.14）
pip install torch>=2.0.0
pip install opencv-python>=4.8.0
pip install ultralytics>=8.0.0
pip install numpy>=1.24.0
```

### 驗證安裝
```bash
python pose_config.py
```

應該顯示：
```
=== 系統配置 ===
Python 版本: 3.14.0
MediaPipe: ✗ 未安裝
YOLOv8: ✓ 可用

當前: 使用 YOLOv8（準確率可能較低）
建議: 安裝 Python 3.11 + MediaPipe 以提升準確率
```

---

## 🚀 使用方式

系統會**自動選擇**可用的姿勢估計器（優先 MediaPipe）。

### 單一影片分類
```bash
python demo.py
# 選擇選項 1，輸入影片檔名
```

### 帶視覺化輸出
```bash
python demo.py
# 選擇選項 2，輸入影片檔名和輸出檔名
```

### 批次處理
```bash
python demo.py
# 選擇選項 3，依序輸入多個影片檔名
```

### 命令列使用
```bash
# 基本預測
python inference_v2.py your_video.mp4

# 預測並產生視覺化影片
python inference_v2.py your_video.mp4 output.mp4
```

---

## 📊 預期效果

### MediaPipe 版本
```
窗口預測詳情:
  窗口 0: 肩上推舉 (信心度: 95.5%)
  窗口 1: 肩上推舉 (信心度: 97.2%)
  窗口 2: 肩上推舉 (信心度: 96.8%)

預測結果: 肩上推舉 (Shoulder Press)
信心度: 96.50%
```

### YOLOv8 版本（較不穩定）
```
窗口預測詳情:
  窗口 0: 槓鈴二頭彎舉 (信心度: 58.06%)
  窗口 1: 肩上推舉 (信心度: 69.86%)
  窗口 2: 深蹲 (信心度: 45.85%)

預測結果: 深蹲 (Squat)
信心度: 50.23%
```

---

## 🔧 診斷工具

### 檢查系統配置
```bash
python pose_config.py
```

### 測試模型輸入
```bash
python debug_model.py example_video.mp4
```

### 分析影片片段
```bash
python analyze_video.py example_video.mp4
```

---

## ❓ 常見問題

### Q1: MediaPipe 安裝失敗
**A:** 確認 Python 版本 ≤ 3.11。MediaPipe 不支援 Python 3.12+。

### Q2: 預測結果不準確
**A:**
1. 確認使用 MediaPipe（準確率更高）
2. 檢查影片是否包含多個動作片段
3. 使用 `analyze_video.py` 分析影片內容

### Q3: 如何在兩個版本間切換
**A:** 系統會自動選擇。如果同時安裝了 MediaPipe 和 YOLOv8，優先使用 MediaPipe。

---

## 📁 專案結構

```
專題/
├── bilstm_mix_best_pt.pth           # 訓練好的模型權重
├── test.py                           # BiLSTM 模型定義
├── pose_config.py                    # 自動配置系統 ⭐
├── pose_extractor_mediapipe.py       # MediaPipe 提取器 ⭐
├── pose_extractor.py                 # YOLOv8 提取器
├── feature_utils_v2.py               # 特徵處理（102 維）
├── inference_v2.py                   # 主要推論系統
├── demo.py                           # 互動式示範
├── debug_model.py                    # 診斷工具
├── analyze_video.py                  # 影片分析工具
├── requirements.txt                  # 套件清單（YOLOv8）
└── SETUP.md                          # 本文件
```

---

## 📞 技術支援

遇到問題？請提供以下資訊：

```bash
# 1. 系統資訊
python pose_config.py

# 2. 診斷輸出
python debug_model.py example_video.mp4 > debug_output.txt

# 3. Python 版本
python --version
```

---

**祝你專題順利！🎓**
