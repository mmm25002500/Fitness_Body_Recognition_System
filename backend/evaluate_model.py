#!/usr/bin/env python3
"""
模型評估腳本
評估訓練好的模型效能
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support
)
import json

from model import BiLSTMAttention
from training_config import MODEL_CONFIG, OUTPUT_PATHS, CLASS_TO_ID
from train_exercise_model import ExerciseDataset


def plot_confusion_matrix(cm, class_names, save_path):
    """繪製混淆矩陣"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.title('Confusion Matrix', fontsize=16, pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 混淆矩陣保存至: {save_path}")


def plot_class_accuracy(class_names, accuracies, save_path):
    """繪製每個類別的準確率"""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_names)), accuracies, color='skyblue')

    # 在柱狀圖上標註數值
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2., height,
            f'{height:.1f}%',
            ha='center', va='bottom'
        )

    plt.xlabel('Exercise Class', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('Per-Class Accuracy', fontsize=16, pad=20)
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha='right')
    plt.ylim(0, 110)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 類別準確率圖保存至: {save_path}")


def evaluate_model():
    """完整評估流程"""
    print("=" * 60)
    print("開始評估模型")
    print("=" * 60)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    # 載入測試資料
    print("\n載入測試資料...")
    data_dir = Path(OUTPUT_PATHS['processed_data'])
    test_data = np.load(data_dir / 'test.npz')
    X_test, y_test = test_data['X'], test_data['y']
    print(f"  測試集: {X_test.shape[0]} 樣本")

    # 建立 DataLoader
    test_dataset = ExerciseDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # 載入模型
    print("\n載入模型...")
    model_path = Path(OUTPUT_PATHS['models']) / 'exercise_model_best.pth'

    if not model_path.exists():
        print(f"❌ 找不到模型: {model_path}")
        print("請先執行 train_exercise_model.py 訓練模型")
        return

    model = BiLSTMAttention(**MODEL_CONFIG).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"  ✓ 模型載入成功: {model_path}")

    # 評估
    print("\n開始評估...")
    all_predictions = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 計算指標
    print("\n" + "=" * 60)
    print("評估結果")
    print("=" * 60)

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"\n總體準確率: {accuracy * 100:.2f}%")

    # 每個類別的指標
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_predictions, average=None
    )

    # 類別名稱
    class_names = list(CLASS_TO_ID.keys())

    print("\n各類別詳細指標:")
    print("-" * 80)
    print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)

    for i, class_name in enumerate(class_names):
        print(f"{class_name:<25} {precision[i]:<12.4f} {recall[i]:<12.4f} "
              f"{f1[i]:<12.4f} {support[i]:<10}")

    print("-" * 80)
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1)
    print(f"{'Average':<25} {avg_precision:<12.4f} {avg_recall:<12.4f} {avg_f1:<12.4f}")

    # 混淆矩陣
    cm = confusion_matrix(all_labels, all_predictions)
    print("\n混淆矩陣:")
    print(cm)

    # 每個類別的準確率
    class_accuracies = []
    for i in range(len(class_names)):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = (all_predictions[class_mask] == i).sum() / class_mask.sum()
            class_accuracies.append(class_acc * 100)
        else:
            class_accuracies.append(0)

    # 準備輸出目錄
    results_dir = Path(OUTPUT_PATHS['results'])
    results_dir.mkdir(exist_ok=True, parents=True)

    # 繪製視覺化
    print("\n生成視覺化圖表...")
    plot_confusion_matrix(cm, class_names, results_dir / 'confusion_matrix.png')
    plot_class_accuracy(class_names, class_accuracies, results_dir / 'class_accuracy.png')

    # 保存詳細報告
    print("\n保存評估報告...")
    report = {
        'overall_accuracy': float(accuracy),
        'per_class_metrics': {
            class_names[i]: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i]),
                'accuracy': float(class_accuracies[i])
            }
            for i in range(len(class_names))
        },
        'average_metrics': {
            'precision': float(avg_precision),
            'recall': float(avg_recall),
            'f1_score': float(avg_f1)
        },
        'confusion_matrix': cm.tolist()
    }

    report_path = results_dir / 'evaluation_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON 報告: {report_path}")

    # 保存文字報告
    text_report_path = results_dir / 'evaluation_report.txt'
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("模型評估報告\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"總體準確率: {accuracy * 100:.2f}%\n\n")
        f.write(classification_report(
            all_labels, all_predictions,
            target_names=class_names,
            digits=4
        ))
        f.write("\n\n混淆矩陣:\n")
        f.write(str(cm))
    print(f"  ✓ 文字報告: {text_report_path}")

    # 找出錯誤分類的樣本
    print("\n分析錯誤分類...")
    misclassified = np.where(all_predictions != all_labels)[0]
    print(f"  錯誤分類樣本數: {len(misclassified)} / {len(all_labels)} ({len(misclassified)/len(all_labels)*100:.2f}%)")

    if len(misclassified) > 0:
        errors_path = results_dir / 'misclassified_samples.txt'
        with open(errors_path, 'w', encoding='utf-8') as f:
            f.write("錯誤分類樣本\n")
            f.write("=" * 60 + "\n\n")

            for idx in misclassified[:50]:  # 只顯示前 50 個
                true_class = class_names[all_labels[idx]]
                pred_class = class_names[all_predictions[idx]]
                confidence = all_probs[idx][all_predictions[idx]]

                f.write(f"樣本 #{idx}\n")
                f.write(f"  真實類別: {true_class}\n")
                f.write(f"  預測類別: {pred_class} (信心度: {confidence:.4f})\n")
                f.write(f"  各類別機率: {all_probs[idx]}\n")
                f.write("\n")

        print(f"  ✓ 錯誤樣本分析: {errors_path}")

    # 總結
    print("\n" + "=" * 60)
    print("✅ 評估完成！")
    print("=" * 60)
    print(f"\n結果保存於: {results_dir}")
    print(f"\n模型準確率: {accuracy * 100:.2f}%")

    # 判斷模型是否可用
    if accuracy > 0.85:
        print("\n✅ 模型表現良好！可以部署使用")
        print("\n下一步:")
        print("1. 將模型複製到 backend/ 目錄")
        print(f"   cp {model_path} ./bilstm_mix_best_pt.pth")
        print("2. 重新啟動 FastAPI backend")
    elif accuracy > 0.70:
        print("\n⚠️  模型表現尚可，但建議改進:")
        print("  - 增加訓練資料")
        print("  - 使用資料增強")
        print("  - 調整模型超參數")
    else:
        print("\n❌ 模型表現較差，建議:")
        print("  - 檢查訓練資料品質")
        print("  - 增加更多樣本")
        print("  - 改進特徵提取")


if __name__ == "__main__":
    # 設置環境變數
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        evaluate_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  評估被中斷")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
