# 🚀 啟動指南 - Next.js + FastAPI 架構

## 📦 架構說明

```
專題/
├── backend/           # Python FastAPI 後端 (Port 8000)
│   ├── main.py       # FastAPI 主程式
│   ├── requirements.txt
│   └── run_backend.sh
│
└── frontend/         # Next.js 前端 (Port 3000)
    ├── app/
    ├── components/
    └── package.json
```

## ⚡ 快速啟動

### 1️⃣ 啟動後端 (Python FastAPI)

```bash
cd backend
bash run_backend.sh
```

後端將運行在 **http://localhost:8000**

### 2️⃣ 啟動前端 (Next.js)

開啟新的終端視窗：

```bash
cd frontend
npm install  # 首次執行需要安裝依賴
npm run dev
```

前端將運行在 **http://localhost:3000**

### 3️⃣ 開啟瀏覽器

訪問 **http://localhost:3000** 開始使用

---

## 🔧 詳細步驟

### 後端設定

1. **確認 Python 環境**
   ```bash
   python --version  # 需要 Python 3.11+
   ```

2. **啟動後端**
   ```bash
   cd backend
   bash run_backend.sh
   ```

3. **測試 API**
   - 開啟 http://localhost:8000
   - 查看 API 文檔：http://localhost:8000/docs

### 前端設定

1. **安裝依賴** (首次執行)
   ```bash
   cd frontend
   npm install
   ```

2. **啟動開發伺服器**
   ```bash
   npm run dev
   ```

3. **開啟應用**
   - 瀏覽器訪問 http://localhost:3000

---

## 📱 使用方式

### 🤖 自動模式
1. 點選「Automatic Mode」
2. 上傳運動影片
3. AI 自動辨識運動類型並計數

### ✋ 手動模式
1. 點選「Manual Mode」
2. 選擇運動類型（深蹲、肩推等）
3. 上傳影片
4. 系統自動計數

---

## 🎯 技術棧

| 層級 | 技術 | 用途 |
|------|------|------|
| **前端** | Next.js 15 + TypeScript | UI/UX 介面 |
| **前端樣式** | Tailwind CSS 4 | 響應式設計 |
| **後端** | Python FastAPI | REST API + WebSocket |
| **AI 模型** | PyTorch BiLSTM | 運動辨識 |
| **姿態估計** | MediaPipe | 骨架提取 |
| **通訊** | WebSocket | 即時影片處理 |

---

## 🐛 除錯

### 後端無法啟動
```bash
# 檢查端口是否被佔用
lsof -i :8000

# 手動安裝依賴
cd backend
pip install -r requirements.txt
```

### 前端無法連接後端
- 確認後端運行在 http://localhost:8000
- 檢查瀏覽器控制台的 WebSocket 錯誤
- 確認 CORS 設定正確

### WebSocket 連接失敗
```bash
# 測試後端是否正常
curl http://localhost:8000/

# 應該返回：
# {"status":"running","model_loaded":true,"pose_extractor_ready":true}
```

---

## 📊 API 端點

| 端點 | 方法 | 用途 |
|------|------|------|
| `/` | GET | 健康檢查 |
| `/api/exercises` | GET | 取得支援的運動列表 |
| `/api/upload` | POST | 上傳影片 |
| `/api/predict` | POST | 預測運動類型 |
| `/ws/process` | WebSocket | 即時影片處理 |

---

## 🎨 前端組件

- **VideoUploader** - 影片上傳介面
- **VideoPlayer** - 即時影片處理與顯示
- **ExerciseSelector** - 運動類型選擇器
- **StatsPanel** - 統計資訊面板

---

## 📦 部署

### 後端 (Python)
- **平台**: Railway / Render / AWS
- **環境變數**: 無需額外設定
- **端口**: 8000

### 前端 (Next.js)
- **平台**: Vercel (推薦)
- **環境變數**:
  ```
  NEXT_PUBLIC_API_URL=https://your-backend.railway.app
  ```
- **端口**: 3000

---

**完整專案說明** → README.md
