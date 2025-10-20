# 📁 專案檔案說明

## ✅ 保留的檔案

### 🌟 核心應用
| 檔案 | 用途 | 必要性 |
|------|------|--------|
| `app.py` | Streamlit Web 主程式 | ⭐⭐⭐ 必須 |
| `exercise_counter.py` | 運動計數模組 | ⭐⭐⭐ 必須 |
| `visualization.py` | 視覺化繪圖工具 | ⭐⭐⭐ 必須 |

### 🧠 AI 模型
| 檔案 | 用途 | 必要性 |
|------|------|--------|
| `model.py` | BiLSTM 模型定義 | ⭐⭐⭐ 必須 |
| `bilstm_mix_best_pt.pth` | 訓練好的模型權重 | ⭐⭐⭐ 必須 |
| `feature_utils_v2.py` | 特徵提取工具 | ⭐⭐⭐ 必須 |

### 📷 姿態估計
| 檔案 | 用途 | 必要性 |
|------|------|--------|
| `pose_extractor_mediapipe.py` | MediaPipe 姿態估計（推薦）| ⭐⭐⭐ 必須 |
| `pose_extractor.py` | YOLOv8 姿態估計（備用）| ⭐ 可選 |
| `pose_config.py` | 自動選擇姿態估計器 | ⭐⭐ 建議 |

### 🛠️ 工具腳本
| 檔案 | 用途 | 必要性 |
|------|------|--------|
| `install_mediapipe.sh` | 自動安裝腳本 | ⭐⭐⭐ 必須 |
| `run_app.sh` | 啟動應用腳本 | ⭐⭐⭐ 必須 |

### 📝 文檔
| 檔案 | 用途 | 必要性 |
|------|------|--------|
| `README.md` | 完整專案說明 | ⭐⭐⭐ 必須 |
| `USAGE.md` | 快速使用指南 | ⭐⭐ 建議 |
| `.gitignore` | Git 排除設定 | ⭐⭐ 建議 |

### 🔧 備用功能
| 檔案 | 用途 | 必要性 |
|------|------|--------|
| `inference_v2.py` | 命令列推論工具 | ⭐ 可選 |

---

## ❌ 已刪除的檔案

### 除錯工具（不再需要）
- ❌ `test_counter.py` - 測試計數器
- ❌ `test_streamlit_counter.py` - 測試 Streamlit
- ❌ `verify_counter.py` - 驗證計數邏輯
- ❌ `debug_model.py` - 模型除錯工具
- ❌ `analyze_video.py` - 影片分析工具

### 舊版本（已被取代）
- ❌ `demo.py` - 舊的命令列示範（被 app.py 取代）
- ❌ `inference.py` - 舊版推論（被 inference_v2.py 取代）
- ❌ `feature_utils.py` - 舊版特徵（被 feature_utils_v2.py 取代）
- ❌ `predict_segment.py` - 片段預測（已整合）

### 舊文檔（已整合）
- ❌ `SETUP.md` - 安裝指南（整合到 README.md）
- ❌ `COUNTER_GUIDE.md` - 計數指南（整合到 USAGE.md）
- ❌ `STREAMLIT_TEST.md` - 測試指南（不再需要）

### 舊腳本
- ❌ `run_with_mediapipe.sh` - 被 run_app.sh 取代

---

## 📊 檔案統計

- **總保留檔案**: 12 個
- **核心必須**: 9 個
- **建議保留**: 2 個
- **可選功能**: 1 個
- **已刪除**: 13 個

---

## 🎯 最小執行環境

只需這 9 個檔案即可運行：

1. `app.py`
2. `exercise_counter.py`
3. `visualization.py`
4. `model.py`
5. `bilstm_mix_best_pt.pth`
6. `feature_utils_v2.py`
7. `pose_extractor_mediapipe.py`
8. `install_mediapipe.sh`
9. `run_app.sh`

---

## 📦 建議保留（完整功能）

加上這些檔案以獲得完整功能：

10. `pose_config.py` - 自動選擇最佳姿態估計器
11. `README.md` - 完整說明文檔
12. `USAGE.md` - 快速入門指南

---

**現在專案結構清晰，只保留真正需要的檔案！** ✨
