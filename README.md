# 健身肢體辨識系統 文檔 - Docs of Fitness Body Recognition System
一個基於 BiLSTM 和 MediaPipe 的健身肢體辨識系統，可支援全自動或全手動識別五種不同的運動模式。

## 介紹

### Skill Stack
* Frontend: Next.js 15, React, TypeScript, Tailwind
* Backend(API): Python FastAPI
* ML: PyTorch BiLSTM with Attention
* Pose Estimation: MediaPipe Pose(33 points)
* API: WebSocket + REST API

### 支援運動
1. 槓鈴二頭彎舉
2. 錘式彎舉
3. 伏地挺身
4. 肩上推舉
5. 深蹲

## 如何安裝
### System Requirements
- Python 3.11+
- Node.js 18+
- pnpm or npm

### Installation
1. 後端必要檔案：`python3.11 -m pip install -r requirements.txt`
2. 後端伺服器: `bash backend/run_backend.sh`
3. 前端安裝: `pnpm i`
4. 前端熱更新：`pnpm dev`

#### 「或」一鍵啟動
```bash
bash start_all.sh
```

預設 port 為 3000，熱更新網頁為 http://localhost:3000

## 系統架構

### 前端 (Next.js + TypeScript + Tailwind CSS)
- port：3000
- 處理使用者介面與影片上傳
- 透過 WebSocket 與後端即時通訊

### 後端 (Python + FastAPI)
- 端口：8000
- BiLSTM 模型進行運動分類
- MediaPipe 進行姿態估計
- 即時影片幀處理

## 專案結構

```
${Project_Folder}/
├── backend/
│   ├── main.py              # FastAPI 應用程式
│   ├── requirements.txt     # Python 依賴套件
│   └── run_backend.sh       # 後端啟動腳本
├── frontend/
│   ├── app/
│   │   └── page.tsx        # 主頁面
│   ├── components/         # React 組件
│   └── package.json
├── model.py                # BiLSTM 模型定義
├── bilstm_mix_best_pt.pth # 訓練好的模型權重
├── exercise_counter.py     # 計數邏輯
├── pose_extractor_mediapipe.py # 姿態估計
├── visualization.py        # 繪圖工具
└── feature_utils_v2.py    # 特徵提取
```

## API
- `GET /` - 健康檢查
- `GET /api/exercises` - 取得支援的運動列表
- `POST /api/upload` - 上傳影片檔案
- `POST /api/predict` - 預測運動類型
- `WS /ws/process` - WebSocket 即時影片處理

## Model Details
- **架構**：雙向 LSTM 搭配注意力機制
- **輸入**：102 維特徵向量（45 幀，步長 3）
- **特徵**：8 個關節角度 + 51 個 3D 座標 + 43 個幾何特徵
- **準確率**：測試集上達 85-90%

## 授權
MIT License