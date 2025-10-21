"""
訓練配置檔案
定義所有訓練相關的超參數和路徑
"""

# 資料集路徑
DATASET_PATH = "../archive/final_kaggle_with_additional_video"

# 運動類別
EXERCISE_CLASSES = [
    "barbell biceps curl",
    "hammer curl",
    "push-up",
    "shoulder press",
    "squat"
]

# 類別映射 (資料夾名稱 -> 類別ID)
CLASS_TO_ID = {
    "barbell biceps curl": 0,
    "hammer curl": 1,
    "push-up": 2,
    "shoulder press": 3,
    "squat": 4
}

# 特徵提取參數
FEATURE_CONFIG = {
    'input_dim': 102,          # 特徵維度
    'window_size': 45,         # 時間窗口大小
    'stride': 3,               # 滑動窗口步幅
    'min_frames': 45,          # 最少需要的幀數
}

# 模型架構參數
MODEL_CONFIG = {
    'input_dim': 102,
    'hidden_dim': 96,
    'attn_dim': 128,
    'num_classes': 5,
}

# 訓練參數
TRAINING_CONFIG = {
    'batch_size': 16,          # 較小的 batch size (因為資料不多)
    'learning_rate': 0.001,
    'weight_decay': 1e-5,      # L2 正則化
    'epochs': 100,             # 最大 epoch 數
    'early_stopping_patience': 15,  # Early stopping 的耐心值
    'val_split': 0.15,         # 驗證集比例
    'test_split': 0.15,        # 測試集比例
}

# 資料增強參數
AUGMENTATION_CONFIG = {
    'enable': True,
    'horizontal_flip_prob': 0.3,    # 水平翻轉機率
    'time_stretch_range': (0.8, 1.2),  # 時間伸縮範圍
    'noise_std': 0.01,              # 高斯噪聲標準差
}

# 輸出路徑
OUTPUT_PATHS = {
    'processed_data': './training_data',
    'models': './models',
    'results': './results',
    'tensorboard': './runs',
}

# 模型保存
MODEL_SAVE = {
    'save_best_only': True,
    'save_interval': 5,  # 每 5 個 epoch 保存一次
    'checkpoint_prefix': 'exercise_model',
}

# 日誌設置
LOGGING = {
    'print_interval': 10,  # 每 10 個 batch 打印一次
    'use_tensorboard': True,
}

# 設備設置
DEVICE = 'cuda'  # 'cuda' 或 'cpu'，會自動檢測
