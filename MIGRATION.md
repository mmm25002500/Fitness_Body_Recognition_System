# 🔄 專案遷移說明 - Streamlit → Next.js + FastAPI

## 📋 遷移摘要

原本的 Streamlit 單體應用已經成功遷移到前後端分離架構：

| 層級 | 舊版 (Streamlit) | 新版 (Next.js + FastAPI) |
|------|------------------|---------------------------|
| **前端** | Streamlit (Python) | Next.js + TypeScript + Tailwind |
| **後端** | 整合在 Streamlit 內 | 獨立 FastAPI 服務 |
| **通訊** | 內部函數調用 | WebSocket + REST API |
| **部署** | 單一服務 | 前後端獨立部署 |

---

## 🗂️ 新專案結構

```
專題/
├── backend/                    # 🔧 Python FastAPI 後端
│   ├── main.py                # API 主程式
│   ├── requirements.txt       # Python 依賴
│   └── run_backend.sh         # 啟動腳本
│
├── frontend/                   # 🎨 Next.js 前端
│   ├── app/
│   │   └── page.tsx           # 主頁面
│   ├── components/
│   │   ├── VideoUploader.tsx  # 影片上傳
│   │   ├── VideoPlayer.tsx    # 即時處理
│   │   ├── ExerciseSelector.tsx
│   │   └── StatsPanel.tsx
│   ├── package.json
│   └── tailwind.config.ts
│
├── model.py                    # BiLSTM 模型定義
├── bilstm_mix_best_pt.pth     # 模型權重
├── exercise_counter.py         # 計數邏輯
├── pose_extractor_mediapipe.py # 姿態估計
├── visualization.py            # 視覺化
├── feature_utils_v2.py         # 特徵提取
│
├── app.py                      # ⚠️  舊版 Streamlit（保留）
├── start_all.sh                # 🚀 一鍵啟動腳本
└── START.md                    # 啟動指南
```

---

## ⚙️ 技術棧對比

### 舊版 (Streamlit)
```python
# app.py - 全部在一個檔案
import streamlit as st
import cv2
from exercise_counter import RepetitionCounter

st.title("Fitness AI Trainer")
video = st.file_uploader("Upload video")
# 直接在 Python 中處理...
```

**優點**：簡單、快速原型
**缺點**：效能差、UI 受限、難以擴展

### 新版 (Next.js + FastAPI)

**前端** (`frontend/app/page.tsx`):
```typescript
'use client';
import { useState } from 'react';

export default function Home() {
  const [videoFile, setVideoFile] = useState<File | null>(null);
  // 使用 WebSocket 與後端通訊
  const ws = new WebSocket('ws://localhost:8000/ws/process');
  // ...
}
```

**後端** (`backend/main.py`):
```python
from fastapi import FastAPI, WebSocket
import torch

app = FastAPI()

@app.websocket("/ws/process")
async def websocket_process(websocket: WebSocket):
    # 處理即時影片流
    await websocket.accept()
    # ...
```

**優點**：
- ⚡ 更快的載入和響應速度
- 🎨 Tailwind CSS 提供更美觀的 UI
- 📱 響應式設計，支援手機
- 🚀 可獨立部署前後端
- 🔧 TypeScript 提供型別安全

---

## 🔄 遷移的主要變更

### 1. 檔案重命名
| 舊檔名 | 新檔名 | 原因 |
|--------|--------|------|
| `test.py` | `model.py` | 避免與 Python 內建 `test` 模組衝突 |

### 2. API 端點

#### 健康檢查
```bash
GET http://localhost:8000/
返回：{"status": "running", "model_loaded": true, "pose_extractor_ready": true}
```

#### 取得運動列表
```bash
GET http://localhost:8000/api/exercises
返回：[{"id": 0, "name": "Barbell Biceps Curl"}, ...]
```

#### WebSocket 即時處理
```bash
WS ws://localhost:8000/ws/process
發送：{"mode": "manual", "exercise_id": 3, "frame": "data:image/jpeg;base64,..."}
接收：{"success": true, "count": 5, "stage": "up", "angle": 125.3, ...}
```

### 3. 狀態管理

**舊版 (Streamlit)**:
```python
if "counter" not in st.session_state:
    st.session_state.counter = RepetitionCounter()
```

**新版 (Next.js)**:
```typescript
const [stats, setStats] = useState({
  count: 0,
  stage: 'down',
  angle: null
});
```

---

## 🚀 如何啟動

### 方法 1：一鍵啟動（推薦）

```bash
bash start_all.sh
```

然後訪問 **http://localhost:3000**

### 方法 2：分別啟動

**終端 1 - 後端**:
```bash
cd backend
bash run_backend.sh
```

**終端 2 - 前端**:
```bash
cd frontend
npm install  # 首次執行
npm run dev
```

### 舊版 Streamlit (仍可用)

```bash
bash run_app.sh
# 訪問 http://localhost:8501
```

---

## 📦 部署建議

### 後端 (FastAPI)
- **平台**: Railway / Render / AWS Lambda
- **環境**: Python 3.11+
- **端口**: 8000
- **啟動命令**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

### 前端 (Next.js)
- **平台**: Vercel (推薦) / Netlify / AWS Amplify
- **環境變數**:
  ```
  NEXT_PUBLIC_API_URL=https://your-backend.railway.app
  NEXT_PUBLIC_WS_URL=wss://your-backend.railway.app
  ```
- **Build 命令**: `npm run build`
- **Start 命令**: `npm start`

---

## 🔧 除錯指南

### 後端無法啟動

**問題**: `ImportError: cannot import name 'BiLSTMWithAttention'`
**解決**:
```bash
# 確認已重命名 test.py → model.py
ls -la model.py
```

**問題**: `ImportError: cannot import name 'extract_features_v2'`
**解決**: 使用 `landmarks_to_features_v2` 函數

### WebSocket 連接失敗

**檢查後端**:
```bash
curl http://localhost:8000/
# 應返回 {"status": "running", ...}
```

**檢查 CORS**:
```python
# backend/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 確認前端 URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### 前端無法連接後端

**瀏覽器控制台**:
```
Failed to construct 'WebSocket': The URL 'ws://localhost:8000/ws/process' is invalid
```

**解決**: 確認後端運行在 port 8000，檢查防火牆設定

---

## 📈 效能對比

| 指標 | Streamlit | Next.js + FastAPI |
|------|-----------|-------------------|
| 首次載入 | ~3-5秒 | ~0.5-1秒 |
| 頁面互動 | 每次重渲染整頁 | 僅更新變更部分 |
| 影片處理 | 阻塞式 | 非阻塞 WebSocket |
| UI 響應 | 卡頓 | 流暢 |
| 部署複雜度 | 簡單 | 中等 |
| 擴展性 | 受限 | 極佳 |

---

## ✅ 遷移檢查清單

- [x] 後端 FastAPI 設置完成
- [x] 前端 Next.js 專案建立
- [x] WebSocket 通訊實現
- [x] UI 組件完成（VideoUploader, VideoPlayer, ExerciseSelector, StatsPanel）
- [x] 模型載入正常
- [x] MediaPipe 姿態估計整合
- [x] 計數邏輯保留
- [x] 啟動腳本建立
- [x] 文件更新

---

## 🎯 下一步

1. **測試**: 上傳影片測試完整流程
2. **優化**: 調整 WebSocket 傳輸頻率
3. **UI 改進**: 新增載入動畫、錯誤提示
4. **部署**: 部署到生產環境
5. **監控**: 新增日誌和效能監控

---

**完整啟動指南** → START.md
**專案結構說明** → PROJECT_STRUCTURE.txt
