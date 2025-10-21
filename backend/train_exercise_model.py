#!/usr/bin/env python3
"""
模型訓練腳本
使用預處理好的資料訓練 BiLSTM 模型
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from model import BiLSTMAttention
from training_config import (
    MODEL_CONFIG, TRAINING_CONFIG, OUTPUT_PATHS, MODEL_SAVE
)


class ExerciseDataset(Dataset):
    """運動資料集"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EarlyStopping:
    """Early stopping 工具"""

    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def train_epoch(model, train_loader, criterion, optimizer, device):
    """訓練一個 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # 統計
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 更新進度條
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """驗證模型"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(val_loader), 100. * correct / total


def train_model():
    """完整訓練流程"""
    print("=" * 60)
    print("開始訓練模型")
    print("=" * 60)

    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用設備: {device}")

    # 載入資料
    print("\n載入訓練資料...")
    data_dir = Path(OUTPUT_PATHS['processed_data'])

    train_data = np.load(data_dir / 'train.npz')
    val_data = np.load(data_dir / 'val.npz')

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']

    print(f"  訓練集: {X_train.shape[0]} 樣本")
    print(f"  驗證集: {X_val.shape[0]} 樣本")

    # 建立 DataLoader
    train_dataset = ExerciseDataset(X_train, y_train)
    val_dataset = ExerciseDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=True,
        num_workers=0  # MacOS 上設為 0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=TRAINING_CONFIG['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # 建立模型
    print("\n建立模型...")
    model = BiLSTMAttention(**MODEL_CONFIG).to(device)
    print(model)

    # 計算模型參數量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n總參數量: {total_params:,}")
    print(f"可訓練參數: {trainable_params:,}")

    # 損失函數和優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )

    # 學習率調度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=TRAINING_CONFIG['early_stopping_patience']
    )

    # TensorBoard
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(f"{OUTPUT_PATHS['tensorboard']}/{timestamp}")

    # 準備輸出目錄
    model_dir = Path(OUTPUT_PATHS['models'])
    model_dir.mkdir(exist_ok=True, parents=True)

    # 訓練迴圈
    print(f"\n開始訓練 (最多 {TRAINING_CONFIG['epochs']} epochs)...")
    print("=" * 60)

    best_val_acc = 0
    best_val_loss = float('inf')
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }

    for epoch in range(TRAINING_CONFIG['epochs']):
        print(f"\nEpoch {epoch + 1}/{TRAINING_CONFIG['epochs']}")
        print("-" * 60)

        # 訓練
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        # 驗證
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        # 學習率調整
        scheduler.step(val_loss)

        # 記錄
        training_history['train_loss'].append(train_loss)
        training_history['train_acc'].append(train_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)

        # TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # 打印結果
        print(f"\n訓練 - Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"驗證 - Loss: {val_loss:.4f} | Acc: {val_acc:.2f}%")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            save_path = model_dir / f'{MODEL_SAVE["checkpoint_prefix"]}_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 保存最佳模型: {save_path} (acc: {val_acc:.2f}%)")

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.should_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch + 1}")
            break

        # 定期保存
        if (epoch + 1) % MODEL_SAVE['save_interval'] == 0:
            save_path = model_dir / f'{MODEL_SAVE["checkpoint_prefix"]}_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ 保存檢查點: {save_path}")

    writer.close()

    # 保存訓練歷史
    history_path = model_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    print(f"\n✓ 保存訓練歷史: {history_path}")

    # 訓練總結
    print("\n" + "=" * 60)
    print("✅ 訓練完成！")
    print("=" * 60)
    print(f"\n最佳驗證準確率: {best_val_acc:.2f}%")
    print(f"最佳驗證損失: {best_val_loss:.4f}")
    best_model_path = model_dir / f"{MODEL_SAVE['checkpoint_prefix']}_best.pth"
    print(f"\n模型保存於: {best_model_path}")
    print("\n下一步: 執行 evaluate_model.py 評估模型")


if __name__ == "__main__":
    # 設置環境變數
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    try:
        train_model()
    except KeyboardInterrupt:
        print("\n\n⚠️  訓練被中斷")
    except Exception as e:
        print(f"\n\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
