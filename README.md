# 💪 Fitness AI Trainer - 智能運動辨識與計數系統

結合深度學習（BiLSTM + Attention）與電腦視覺技術，提供**自動運動辨識**和**即時計數**功能的智能健身教練系統。

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.14-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.50-red)

---

## ✨ 主要功能

### 🤖 自動模式
- ✅ **AI 自動辨識運動類型**（準確率 85-90%）
- ✅ **即時動作計數**（使用角度閾值演算法）
- ✅ **多類別機率顯示**（了解 AI 的判斷依據）
- ✅ **信心度指標**（評估預測可靠性）

### ✋ 手動模式
- ✅ **手動選擇運動類型**
- ✅ **專注單一動作訓練**
- ✅ **精確計數反饋**

### 📊 視覺化功能
- ✅ **即時骨架疊加**（MediaPipe Pose）
- ✅ **關鍵關節角度顯示**
- ✅ **動作階段標示**（UP/DOWN）
- ✅ **計數器面板**

### 🎯 支援運動類型
1. 槓鈴二頭彎舉 (Barbell Biceps Curl)
2. 錘式彎舉 (Hammer Curl)
3. 伏地挺身 (Push-up)
4. 肩上推舉 (Shoulder Press)
5. 深蹲 (Squat)

---

## 🚀 快速開始

### 方法 1: 自動安裝腳本（推薦）

```bash
# 執行自動安裝腳本
bash install_mediapipe.sh

# 啟動 Web 應用
bash run_app.sh
```

然後在瀏覽器開啟 http://localhost:8501

### 方法 2: 手動安裝

#### 步驟 1: 建立 Python 3.11 虛擬環境

```bash
# 使用系統的 Python 3.11
python3.11 -m venv venv_mediapipe

# 啟動虛擬環境
source venv_mediapipe/bin/activate
```

#### 步驟 2: 安裝套件

```bash
pip install --upgrade pip
pip install torch opencv-python mediapipe==0.10.14 numpy streamlit
```

#### 步驟 3: 驗證安裝

```bash
python pose_config.py
```

應該看到：
```
=== 系統配置 ===
✓ 使用 MediaPipe 0.10.14
Python 版本: 3.11.x
```

#### 步驟 4: 啟動應用

```bash
streamlit run app.py
```

---

## 📖 使用指南

### Web 介面操作

1. **選擇模式**
   - 🤖 自動模式：AI 自動辨識運動
   - ✋ 手動模式：手動選擇運動類型

2. **選擇輸入來源**
   - 📹 影片檔案：上傳運動影片（支援 mp4, avi, mov）
   - 📷 即時攝影機：使用電腦攝影機即時分析

3. **開始訓練**
   - 上傳影片或啟動攝影機
   - 系統會自動偵測姿態並計數
   - 查看即時反饋和統計資料

### 命令列使用（傳統方式）

#### 單一影片分類
```bash
python demo.py
# 選擇選項 1，輸入影片檔名
```

#### 帶視覺化輸出
```bash
python inference_v2.py example_video.mp4 output.mp4
```

---

## 🏗️ 系統架構

```
┌─────────────────┐
│   影片/攝影機    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ MediaPipe Pose  │ ← 33 關節點 (x, y, z, visibility)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  特徵提取器      │ ← 102 維特徵（角度+座標+幾何）
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BiLSTM + Attn   │ ← 深度學習模型
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  運動分類結果    │ ← 5 種運動類別
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  計數演算法      │ ← 角度閾值檢測
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  視覺化輸出      │ ← Streamlit UI
└─────────────────┘
```

---

## 🧠 技術細節

### 深度學習模型
- **架構**: BiLSTM (2層) + Attention Mechanism
- **輸入**: 102 維特徵向量 × 45 幀
- **輸出**: 5 類運動分類
- **訓練數據**: 多種運動影片數據集

### 姿態估計
- **引擎**: Google MediaPipe Pose
- **關節點**: 33 個全身關鍵點
- **座標**: 3D (x, y, z) + 可見度

### 特徵工程（102 維）
1. **8 個關節角度** - 左右手肘、肩膀、膝蓋、髖部
2. **51 個 3D 座標** - 17 個關鍵點的 (x, y, z)
3. **43 個幾何特徵** - 距離、比例、相對位置

### 計數演算法
使用狀態機 + 角度閾值：

| 運動類型 | 關節 | DOWN 閾值 | UP 閾值 |
|---------|------|-----------|---------|
| 槓鈴二頭彎舉 | 手肘 | > 140° | < 90° |
| 錘式彎舉 | 手肘 | > 140° | < 90° |
| 伏地挺身 | 手肘 | < 90° | > 140° |
| 肩上推舉 | 手肘 | < 90° | > 160° |
| 深蹲 | 膝蓋 | > 160° | < 100° |

---

## 📁 專案結構

```
專題/
├── app.py                           # 🌟 Streamlit Web 應用（主程式）
├── exercise_counter.py              # 🌟 運動計數模組
├── visualization.py                 # 🌟 視覺化繪圖模組
├── bilstm_mix_best_pt.pth          # 訓練好的模型權重
├── test.py                          # BiLSTM 模型定義
├── pose_config.py                   # 自動配置系統
├── pose_extractor_mediapipe.py      # MediaPipe 姿態提取器
├── pose_extractor.py                # YOLOv8 姿態提取器（備用）
├── feature_utils_v2.py              # 特徵處理工具
├── inference_v2.py                  # 推論引擎
├── demo.py                          # 命令列示範
├── install_mediapipe.sh             # 自動安裝腳本
├── run_app.sh                       # 🌟 啟動腳本
├── SETUP.md                         # 詳細安裝指南
└── README.md                        # 本文件
```

---

## 📊 效能表現

### 準確率
- **MediaPipe + BiLSTM**: 85-90%
- **YOLOv8 + BiLSTM**: 60-70%

### 速度
- **影片處理**: ~15-20 FPS (CPU)
- **即時攝影機**: ~10-15 FPS (CPU)

### 系統需求
- **RAM**: 建議 8GB+
- **CPU**: 現代多核心處理器
- **GPU**: 非必要（但可加速）

---

## 💡 使用技巧

### 獲得最佳效果
1. ✅ 確保**全身**都在畫面中
2. ✅ 保持良好的**光線**條件
3. ✅ 動作**完整清晰**，避免遮擋
4. ✅ 穿著**對比明顯**的服裝
5. ✅ 避免**快速移動**造成模糊

### 常見問題

**Q: 計數不準確怎麼辦？**
- 確認動作幅度足夠大
- 檢查關節點是否正確偵測
- 調整閾值參數（在 `exercise_counter.py`）

**Q: 自動辨識錯誤？**
- 切換到手動模式
- 確保影片中只有單一運動類型
- 檢查姿態偵測品質

**Q: 即時攝影機延遲？**
- 降低影片解析度
- 關閉其他占用 CPU 的程式
- 考慮使用 GPU 加速

---

## 🔧 開發與自訂

### 新增運動類型

1. **訓練新模型**（修改 `test.py` 中的 `num_classes`）
2. **設定計數閾值**（修改 `exercise_counter.py` 中的 `thresholds`）
3. **更新類別名稱**（修改 `app.py` 中的 `class_names`）

### 調整計數參數

編輯 `exercise_counter.py`:
```python
self.thresholds = {
    0: {"down_angle": 140, "up_angle": 90, "joint": "elbow"},
    # 調整閾值以符合你的需求
}
```

---

## 📚 參考資源

- [MediaPipe Pose](https://google.github.io/mediapipe/solutions/pose.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## 📄 授權

本專案僅供教育和研究用途。

---

## 👥 貢獻者

專題製作團隊

---

## 📞 技術支援

遇到問題？請提供以下資訊：

```bash
# 1. 系統資訊
python pose_config.py

# 2. Python 版本
python --version

# 3. 已安裝套件
pip list | grep -E "mediapipe|streamlit|torch"
```

---

**祝你訓練愉快！💪🏋️‍♂️**
