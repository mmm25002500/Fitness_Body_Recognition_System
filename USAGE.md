# 💪 Fitness AI Trainer - 使用指南

## 🚀 快速開始

### 1. 安裝環境
```bash
bash install_mediapipe.sh
```

### 2. 啟動應用
```bash
bash run_app.sh
```

然後在瀏覽器開啟：http://localhost:8501

---

## 📖 使用方式

### 🤖 自動模式（AI 辨識）
1. 選擇「自動模式」
2. 上傳影片
3. AI 自動辨識運動並計數

**特點：**
- ✅ 自動辨識 5 種運動
- ✅ 顯示信心度和機率
- ✅ 自動計數

### ✋ 手動模式（指定運動）
1. 選擇「手動模式」
2. 選擇運動類型（肩推、深蹲等）
3. 上傳影片
4. 系統自動計數

**特點：**
- ✅ 專注單一運動訓練
- ✅ 更精確的計數

---

## 🎯 支援運動

1. 槓鈴二頭彎舉 (Barbell Biceps Curl)
2. 錘式彎舉 (Hammer Curl)
3. 伏地挺身 (Push-up)
4. 肩上推舉 (Shoulder Press)
5. 深蹲 (Squat)

---

## 💡 使用技巧

### 獲得最佳計數效果
- ✅ 全身在畫面中
- ✅ 充足光線
- ✅ 動作完整清晰
- ✅ 避免快速移動

### 重置計數
點擊側邊欄的「🔄 重置計數器」

### 除錯模式
勾選「🐛 除錯模式」查看即時角度和計數資訊

---

## 📁 專案結構

```
專題/
├── app.py                          # 🌟 Streamlit Web 應用（主程式）
├── exercise_counter.py             # 運動計數模組
├── visualization.py                # 視覺化繪圖
├── pose_extractor_mediapipe.py     # MediaPipe 姿態估計
├── test.py                         # BiLSTM 模型定義
├── bilstm_mix_best_pt.pth         # 訓練好的模型
├── install_mediapipe.sh            # 安裝腳本
├── run_app.sh                      # 啟動腳本
└── README.md                       # 完整說明
```

---

## ❓ 常見問題

**Q: 計數不準確？**
- 啟用除錯模式查看角度
- 確保動作幅度足夠大
- 切換到手動模式指定運動類型

**Q: 安裝失敗？**
- 確認 Python 版本 3.11
- 查看 README.md 完整安裝指南

**Q: 計數為 0？**
- 檢查動作幅度
- 確認姿態偵測正常（綠色骨架）
- 系統會自動調整閾值適應小幅度動作

---

## 📊 技術規格

- **姿態估計**: MediaPipe Pose (33 關節點)
- **深度學習**: BiLSTM + Attention
- **計數方式**: 角度閾值 + 動態適應
- **準確率**: 85-90%

---

**完整技術文檔請參考 README.md**
