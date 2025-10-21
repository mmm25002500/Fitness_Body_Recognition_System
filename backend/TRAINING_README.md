# 運動辨識模型訓練指南

## 📊 資料集資訊

- **位置**: `../archive/final_kaggle_with_additional_video/`
- **類別**: 5 種運動
  - Barbell Biceps Curl (25 影片)
  - Hammer Curl (12 影片)
  - Push-up (25 影片)
  - Shoulder Press (20 影片)
  - Squat (19 影片)
- **總計**: 101 個訓練影片

## 🚀 快速開始

### 方法 1: 一鍵執行完整訓練

```bash
cd /Users/tershi/Project/專題/backend
./run_full_training.sh
```

這個腳本會依序執行：
1. 資料預處理
2. 模型訓練
3. 模型評估

### 方法 2: 分步執行

#### 步驟 1: 資料預處理 (約 20-30 分鐘)

```bash
../venv_mediapipe/bin/python3 prepare_training_data.py
```

**輸出**:
- `training_data/train.npz` - 訓練集
- `training_data/val.npz` - 驗證集
- `training_data/test.npz` - 測試集
- `training_data/dataset_info.json` - 資料集資訊

#### 步驟 2: 訓練模型 (約 1-2 小時 CPU / 15-30 分鐘 GPU)

```bash
../venv_mediapipe/bin/python3 train_exercise_model.py
```

**輸出**:
- `models/exercise_model_best.pth` - 最佳模型
- `models/training_history.json` - 訓練歷史
- `runs/` - TensorBoard 日誌

**監控訓練**:
```bash
../venv_mediapipe/bin/tensorboard --logdir runs/
```
然後在瀏覽器開啟 http://localhost:6006

#### 步驟 3: 評估模型 (約 5 分鐘)

```bash
../venv_mediapipe/bin/python3 evaluate_model.py
```

**輸出**:
- `results/evaluation_report.json` - JSON 格式評估報告
- `results/evaluation_report.txt` - 文字格式評估報告
- `results/confusion_matrix.png` - 混淆矩陣圖
- `results/class_accuracy.png` - 各類別準確率圖
- `results/misclassified_samples.txt` - 錯誤分類樣本分析

## 📁 訓練產生的檔案結構

```
backend/
├── training_data/          # 預處理後的資料
│   ├── train.npz
│   ├── val.npz
│   ├── test.npz
│   └── dataset_info.json
├── models/                 # 訓練的模型
│   ├── exercise_model_best.pth
│   ├── exercise_model_epoch_*.pth
│   └── training_history.json
├── results/                # 評估結果
│   ├── evaluation_report.json
│   ├── evaluation_report.txt
│   ├── confusion_matrix.png
│   ├── class_accuracy.png
│   └── misclassified_samples.txt
└── runs/                   # TensorBoard 日誌
    └── [timestamp]/
```

## ⚙️ 訓練配置

所有配置在 `training_config.py` 中：

### 模型架構
- **輸入維度**: 102 (特徵維度)
- **Hidden 維度**: 96
- **Attention 維度**: 128
- **輸出類別**: 5

### 訓練參數
- **Batch size**: 16
- **Learning rate**: 0.001
- **Epochs**: 最多 100 (含 early stopping)
- **Early stopping patience**: 15 epochs
- **資料分割**: 訓練 70% / 驗證 15% / 測試 15%

## 🔧 調整訓練參數

編輯 `training_config.py` 來調整：

```python
TRAINING_CONFIG = {
    'batch_size': 16,           # 可調整為 8, 16, 32
    'learning_rate': 0.001,     # 可嘗試 0.0001 - 0.01
    'epochs': 100,              # 最大訓練輪數
    'early_stopping_patience': 15,  # 提前停止的耐心值
}
```

## 📈 提升模型準確率的方法

### 1. 增加訓練資料
- 收集更多影片樣本
- 特別是 Hammer Curl (目前只有 12 個)

### 2. 資料增強
啟用資料增強 (在 `training_config.py`):
```python
AUGMENTATION_CONFIG = {
    'enable': True,  # 設為 True
}
```

### 3. 調整模型架構
在 `training_config.py` 修改:
```python
MODEL_CONFIG = {
    'hidden_dim': 128,  # 增加到 128
    'attn_dim': 256,    # 增加到 256
}
```

### 4. 改進特徵工程
編輯 `feature_utils_v2.py` 添加更多特徵，特別針對深蹲的特徵。

## 🎯 部署訓練好的模型

評估模型後，如果準確率滿意（建議 > 85%），部署模型：

```bash
# 備份舊模型
mv bilstm_mix_best_pt.pth bilstm_mix_best_pt.pth.backup

# 部署新模型
cp models/exercise_model_best.pth bilstm_mix_best_pt.pth

# 重新啟動 backend
../venv_mediapipe/bin/python3 main.py
```

## 🐛 常見問題

### Q: 資料預處理很慢
A: 這是正常的，處理 101 個影片大約需要 20-30 分鐘。可以在背景執行。

### Q: 訓練時記憶體不足
A: 降低 batch_size，例如從 16 改為 8。

### Q: 模型準確率很低
A:
1. 檢查資料品質
2. 增加訓練樣本數
3. 啟用資料增強
4. 調整學習率

### Q: 深蹲辨識不準
A:
1. 增加深蹲訓練樣本
2. 改進特徵提取，添加更多深蹲相關特徵
3. 檢查深蹲影片的拍攝角度

## 📊 查看訓練進度

### 方法 1: 終端輸出
訓練過程會即時顯示 loss 和 accuracy

### 方法 2: TensorBoard
```bash
../venv_mediapipe/bin/tensorboard --logdir runs/
```

在瀏覽器開啟 http://localhost:6006 查看：
- Loss 曲線
- Accuracy 曲線
- Learning rate 變化

## ✅ 驗證模型效果

訓練完成後，使用測試影片驗證：

```bash
# 啟動 backend
../venv_mediapipe/bin/python3 main.py

# 在前端上傳測試影片
# 檢查辨識結果是否正確
```

## 📝 訓練日誌

所有訓練過程都會記錄在：
- 終端輸出
- `models/training_history.json`
- TensorBoard 日誌 (`runs/`)

## 🔄 重新訓練

如果要重新訓練：

```bash
# 刪除舊的訓練資料
rm -rf training_data/ models/ results/ runs/

# 重新執行訓練流程
./run_full_training.sh
```

---

**祝訓練順利！** 🎉

如有問題，請查看評估報告或錯誤日誌。
