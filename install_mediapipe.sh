#!/bin/bash
# MediaPipe 安裝腳本

echo "=== MediaPipe 安裝腳本 ==="
echo ""

# 檢查虛擬環境
if [ ! -d "venv_mediapipe" ]; then
    echo "創建虛擬環境..."
    python3.11 -m venv venv_mediapipe
fi

# 啟動虛擬環境
echo "啟動虛擬環境..."
source venv_mediapipe/bin/activate

echo "Python 版本: $(python --version)"
echo ""

# 安裝套件
echo "安裝套件（這可能需要幾分鐘）..."
pip install --upgrade pip -q
pip install torch opencv-python mediapipe==0.10.14 numpy

echo ""
echo "=== 安裝完成 ==="
echo ""

# 驗證安裝
echo "驗證安裝..."
python pose_config.py

echo ""
echo "使用方式:"
echo "  source venv_mediapipe/bin/activate"
echo "  python demo.py"
echo ""
echo "或使用快捷腳本:"
echo "  ./run_with_mediapipe.sh demo.py"
