# 健身肢體辨識系統 文檔 - Docs of Fitness Body Recognition System
一個基於 BiLSTM 和 MediaPipe 的健身肢體辨識系統，自動識別五種不同的運動模式並提供即時角度監測與警示。

## 介紹

### Skill Stack
* Frontend: Next.js 15, React 19, TypeScript, Tailwind CSS
* Backend(API): Python FastAPI, WebSocket
* ML: PyTorch BiLSTM with Attention
* Pose Estimation: MediaPipe Pose (33 landmarks)
* API: WebSocket + REST API

### 支援運動
1. 槓鈴二頭彎舉 (Barbell Bicep Curl)
2. 錘式彎舉 (Hammer Curl)
3. 伏地挺身 (Push-up)
4. 肩上推舉 (Shoulder Press)
5. 深蹲 (Squat)

### 功能特色
- **自動運動識別**：使用 BiLSTM 模型自動判斷運動類型
- **即時計數追蹤**：自動計算運動次數與階段（上升/下降）
- **角度監測與警示**：
  - 深蹲：角度 <90° 注意、<50° 報警
  - 肩上推舉：角度 <90° 注意、<75° 報警
  - 槓鈴二頭彎舉：角度 <140° 注意
- **即時視覺化**：在影片上標示骨架關節點與角度資訊

## 如何安裝
### System Requirements
- Python 3.11+
- Node.js 20+
- pnpm (推薦) or npm

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
├── backend/                        # 後端核心程式
│   ├── main.py                    # FastAPI 應用程式
│   ├── model.py                   # BiLSTM 模型定義
│   ├── bilstm_mix_best_pt.pth    # 訓練好的模型權重
│   ├── exercise_counter.py        # 計數邏輯
│   ├── pose_extractor_mediapipe.py # 姿態估計
│   ├── visualization.py           # 繪圖工具
│   ├── feature_utils_v2.py       # 特徵提取
│   ├── requirements.txt           # Python 依賴套件
│   └── run_backend.sh            # 後端啟動腳本
├── frontend/                       # 前端核心程式
│   ├── src/
│   │   ├── app/
│   │   │   ├── page.tsx          # 主頁面
│   │   │   ├── layout.tsx        # 全域佈局
│   │   │   └── not-found.tsx     # 404 頁面
│   │   ├── components/           # React 組件
│   │   │   ├── HomeClient.tsx    # 主頁客戶端組件
│   │   │   ├── VideoUploader.tsx # 影片上傳器
│   │   │   ├── VideoPlayer.tsx   # 影片播放器與處理
│   │   │   ├── StatsPanel.tsx    # 統計面板（含角度警示）
│   │   │   ├── Layout/
│   │   │   │   └── Navbar.tsx    # 導航列
│   │   │   └── Footer/
│   │   │       └── Footer.tsx    # 頁尾
│   │   ├── types/                # TypeScript 類型定義
│   │   └── config/               # 配置檔案
│   ├── package.json
│   └── tailwind.config.ts
├── tools/                          # 輔助工具
│   ├── app.py                    # Streamlit 版本（舊版）
│   ├── inference_v2.py           # 命令列推論工具
│   ├── pose_config.py            # 姿態估計器配置
│   └── pose_extractor.py         # YOLOv8 姿態估計（備用）
├── start_all.sh                   # 一鍵啟動腳本
└── README.md                      # 專案文檔
```

## API

### REST API
- `GET /` - 健康檢查
- `GET /api/exercises` - 取得支援的運動列表

### WebSocket API
- `WS /ws/process` - 即時影片處理

#### WebSocket 訊息格式
**發送（Client → Server）：**
```json
{
  "mode": "automatic",
  "exercise_id": 3,
  "frame": "data:image/jpeg;base64,...",
  "debug": false
}
```

**接收（Server → Client）：**
```json
{
  "success": true,
  "frame": "data:image/jpeg;base64,...",
  "count": 5,
  "stage": "up",
  "angle": 85.5,
  "exercise_name": "Squat",
  "predicted_exercise_id": 4,
  "predicted_exercise_name": "Squat",
  "prediction_confidence": 0.92,
  "total_predictions": 15,
  "is_prediction_final": true,
  "warning": "注意角度"
}
```

## Model Details

### BiLSTM 模型
- **架構**：雙向 LSTM 搭配注意力機制
- **輸入**：102 維特徵向量（45 幀，步長 3）
- **特徵**：
  - 8 個關節角度
  - 51 個 3D 座標（17 個關節點 × 3）
  - 43 個幾何特徵
- **準確率**：測試集上達 85-90%

### 姿態估計
- **工具**：MediaPipe Pose
- **關節點**：33 個 landmarks
- **追蹤**：即時骨架偵測與角度計算

### 運動計數邏輯
- 基於關節角度的狀態機
- 檢測動作的「上升」和「下降」階段
- 完成一個完整循環時計數 +1

## 部署

### Cloudflare Pages
前端已部署至 Cloudflare Pages，使用 pnpm 作為套件管理工具。

**部署設定：**
- Build command: `pnpm run build`
- Build output directory: `frontend/.next`
- Root directory: `frontend`
- Node.js version: 20

### 本地開發
後端需在本地執行（WebSocket 連接 localhost:8000）

## 授權
MIT License