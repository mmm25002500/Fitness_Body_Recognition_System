#!/bin/bash
# 使用 MediaPipe 環境執行程式

# 啟動虛擬環境
source venv_mediapipe/bin/activate

# 檢查套件是否安裝
python -c "import mediapipe" 2>/dev/null
if [ $# -eq 0 ]; then
    echo "MediaPipe 虛擬環境已啟動"
    echo "Python 版本: $(python --version)"
    echo ""
    echo "使用方式:"
    echo "  ./run_with_mediapipe.sh demo.py"
    echo "  ./run_with_mediapipe.sh pose_config.py"
    echo "  ./run_with_mediapipe.sh inference_v2.py example_video.mp4"
    echo ""
    exec bash
else
    # 執行傳入的命令
    python "$@"
fi
