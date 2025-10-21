#!/bin/bash
# 完整訓練流程腳本

echo "==========================================================="
echo "  運動辨識模型訓練流程"
echo "==========================================================="
echo ""

# 設定 Python 路徑
PYTHON="../venv_mediapipe/bin/python3"

# 步驟 1: 資料預處理
echo "步驟 1/3: 資料預處理"
echo "-----------------------------------------------------------"
$PYTHON prepare_training_data.py
if [ $? -ne 0 ]; then
    echo "❌ 資料預處理失敗"
    exit 1
fi
echo ""

# 步驟 2: 訓練模型
echo "步驟 2/3: 訓練模型"
echo "-----------------------------------------------------------"
$PYTHON train_exercise_model.py
if [ $? -ne 0 ]; then
    echo "❌ 模型訓練失敗"
    exit 1
fi
echo ""

# 步驟 3: 評估模型
echo "步驟 3/3: 評估模型"
echo "-----------------------------------------------------------"
$PYTHON evaluate_model.py
if [ $? -ne 0 ]; then
    echo "❌ 模型評估失敗"
    exit 1
fi
echo ""

# 完成
echo "==========================================================="
echo "✅ 訓練流程完成！"
echo "==========================================================="
echo ""
echo "下一步:"
echo "1. 查看評估結果: results/evaluation_report.txt"
echo "2. 查看混淆矩陣: results/confusion_matrix.png"
echo "3. 部署模型:"
echo "   cp models/exercise_model_best.pth bilstm_mix_best_pt.pth"
echo ""
