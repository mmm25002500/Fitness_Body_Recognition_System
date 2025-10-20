#!/bin/bash
# Fitness AI Trainer 啟動腳本

echo "🚀 正在啟動 Fitness AI Trainer..."
echo ""

# 啟動虛擬環境
source venv_mediapipe/bin/activate

# 啟動 Streamlit（不自動開啟瀏覽器）
streamlit run app.py --server.port 8501 --server.address localhost --server.headless true

# 說明
echo ""
echo "✓ 應用已啟動！"
echo "📱 請在瀏覽器手動開啟: http://localhost:8501"
echo ""
echo "按 Ctrl+C 停止應用"
