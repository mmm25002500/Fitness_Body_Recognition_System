#!/usr/bin/env python3
"""
將 Keras .h5 模型轉換為 PyTorch .pth 模型
"""
import numpy as np
import torch
import torch.nn as nn
import h5py
import sys

# PyTorch 模型定義 - 匹配 Keras 模型架構
class BiLSTMSimple(nn.Module):
    """簡化版 BiLSTM 模型 - 匹配 Keras .h5 模型架構"""
    def __init__(self, input_dim=22, hidden_dim=91, num_classes=4):
        super().__init__()
        # 第一層 BiLSTM
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(0.5)

        # 第二層 BiLSTM
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(0.5)

        # 輸出層 - 直接從 BiLSTM 的最後一個時間步輸出
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # LSTM 1
        out, _ = self.lstm1(x)
        out = self.dropout1(out)

        # LSTM 2
        out, _ = self.lstm2(out)
        out = self.dropout2(out)

        # 取最後一個時間步
        out = out[:, -1, :]  # (batch, hidden*2)

        # 分類
        out = self.fc(out)
        return out

# 保留原來的複雜模型定義 (用於 backend)
class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, attn_dim)
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=1)
        context = torch.sum(weights * x, dim=1)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=102, hidden_dim=96, attn_dim=128, num_classes=5):
        super().__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.ln1 = nn.LayerNorm(hidden_dim * 2)

        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True)
        self.ln2 = nn.LayerNorm(hidden_dim * 2)

        self.attn = Attention(hidden_dim, attn_dim)

        self.fc1 = nn.Linear(hidden_dim * 2, attn_dim)
        self.fc2 = nn.Linear(attn_dim, num_classes)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.ln1(out)

        out, _ = self.lstm2(out)
        out = self.ln2(out)

        context = self.attn(out)

        out = torch.relu(self.fc1(context))
        out = self.fc2(out)
        return out

def load_keras_weights(h5_path):
    """從 h5 文件載入權重"""
    weights_dict = {}

    with h5py.File(h5_path, 'r') as f:
        print("\n=== Keras 模型權重 ===\n")

        model_weights = f['model_weights']

        # BiLSTM Layer 1 (bidirectional_10)
        print("Layer: bidirectional_10 (第一層 BiLSTM)")
        bi10 = model_weights['bidirectional_10/sequential_5/bidirectional_10']

        # Forward LSTM 1
        fwd_lstm10 = bi10['forward_lstm_10/lstm_cell']
        fwd_kernel_10 = fwd_lstm10['kernel'][:]
        fwd_recurrent_10 = fwd_lstm10['recurrent_kernel'][:]
        fwd_bias_10 = fwd_lstm10['bias'][:]
        print(f"  forward_lstm_10/kernel: {fwd_kernel_10.shape}")
        print(f"  forward_lstm_10/recurrent_kernel: {fwd_recurrent_10.shape}")
        print(f"  forward_lstm_10/bias: {fwd_bias_10.shape}")

        # Backward LSTM 1
        bwd_lstm10 = bi10['backward_lstm_10/lstm_cell']
        bwd_kernel_10 = bwd_lstm10['kernel'][:]
        bwd_recurrent_10 = bwd_lstm10['recurrent_kernel'][:]
        bwd_bias_10 = bwd_lstm10['bias'][:]
        print(f"  backward_lstm_10/kernel: {bwd_kernel_10.shape}")
        print(f"  backward_lstm_10/recurrent_kernel: {bwd_recurrent_10.shape}")
        print(f"  backward_lstm_10/bias: {bwd_bias_10.shape}")

        # BiLSTM Layer 2 (bidirectional_11)
        print("\nLayer: bidirectional_11 (第二層 BiLSTM)")
        bi11 = model_weights['bidirectional_11/sequential_5/bidirectional_11']

        # Forward LSTM 2
        fwd_lstm11 = bi11['forward_lstm_11/lstm_cell']
        fwd_kernel_11 = fwd_lstm11['kernel'][:]
        fwd_recurrent_11 = fwd_lstm11['recurrent_kernel'][:]
        fwd_bias_11 = fwd_lstm11['bias'][:]
        print(f"  forward_lstm_11/kernel: {fwd_kernel_11.shape}")
        print(f"  forward_lstm_11/recurrent_kernel: {fwd_recurrent_11.shape}")
        print(f"  forward_lstm_11/bias: {fwd_bias_11.shape}")

        # Backward LSTM 2
        bwd_lstm11 = bi11['backward_lstm_11/lstm_cell']
        bwd_kernel_11 = bwd_lstm11['kernel'][:]
        bwd_recurrent_11 = bwd_lstm11['recurrent_kernel'][:]
        bwd_bias_11 = bwd_lstm11['bias'][:]
        print(f"  backward_lstm_11/kernel: {bwd_kernel_11.shape}")
        print(f"  backward_lstm_11/recurrent_kernel: {bwd_recurrent_11.shape}")
        print(f"  backward_lstm_11/bias: {bwd_bias_11.shape}")

        # Dense Layer (dense_5)
        print("\nLayer: dense_5 (全連接層)")
        dense5 = model_weights['dense_5/sequential_5/dense_5']
        dense_kernel = dense5['kernel'][:]
        dense_bias = dense5['bias'][:]
        print(f"  dense_5/kernel: {dense_kernel.shape}")
        print(f"  dense_5/bias: {dense_bias.shape}")

        weights_dict = {
            'lstm1': {
                'fwd_kernel': fwd_kernel_10,
                'fwd_recurrent': fwd_recurrent_10,
                'fwd_bias': fwd_bias_10,
                'bwd_kernel': bwd_kernel_10,
                'bwd_recurrent': bwd_recurrent_10,
                'bwd_bias': bwd_bias_10,
            },
            'lstm2': {
                'fwd_kernel': fwd_kernel_11,
                'fwd_recurrent': fwd_recurrent_11,
                'fwd_bias': fwd_bias_11,
                'bwd_kernel': bwd_kernel_11,
                'bwd_recurrent': bwd_recurrent_11,
                'bwd_bias': bwd_bias_11,
            },
            'dense': {
                'kernel': dense_kernel,
                'bias': dense_bias,
            }
        }

    return weights_dict

def convert_lstm_weights(keras_weights, pytorch_lstm, layer_name):
    """
    轉換 Keras BiLSTM 權重到 PyTorch

    Keras LSTM 順序: [input, forget, cell, output] (ifco)
    PyTorch LSTM 順序: [input, forget, cell, output] (ifco) - 相同！
    """
    print(f"\n轉換 {layer_name} 權重...")

    fwd_kernel = keras_weights['fwd_kernel']  # (input_dim, 4*hidden_dim)
    fwd_recurrent = keras_weights['fwd_recurrent']  # (hidden_dim, 4*hidden_dim)
    fwd_bias = keras_weights['fwd_bias']  # (4*hidden_dim,)

    bwd_kernel = keras_weights['bwd_kernel']
    bwd_recurrent = keras_weights['bwd_recurrent']
    bwd_bias = keras_weights['bwd_bias']

    # PyTorch LSTM 參數
    # weight_ih_l0: input-hidden weights (4*hidden_dim, input_dim)
    # weight_hh_l0: hidden-hidden weights (4*hidden_dim, hidden_dim)
    # bias_ih_l0, bias_hh_l0: biases (4*hidden_dim,)

    # Forward direction
    pytorch_lstm.weight_ih_l0.data = torch.FloatTensor(fwd_kernel.T)
    pytorch_lstm.weight_hh_l0.data = torch.FloatTensor(fwd_recurrent.T)
    # Keras 把 bias 合併，PyTorch 分成兩個，我們只設置一個，另一個設為 0
    pytorch_lstm.bias_ih_l0.data = torch.FloatTensor(fwd_bias)
    pytorch_lstm.bias_hh_l0.data = torch.zeros_like(pytorch_lstm.bias_hh_l0.data)

    # Backward direction (reverse 對應的參數名)
    pytorch_lstm.weight_ih_l0_reverse.data = torch.FloatTensor(bwd_kernel.T)
    pytorch_lstm.weight_hh_l0_reverse.data = torch.FloatTensor(bwd_recurrent.T)
    pytorch_lstm.bias_ih_l0_reverse.data = torch.FloatTensor(bwd_bias)
    pytorch_lstm.bias_hh_l0_reverse.data = torch.zeros_like(pytorch_lstm.bias_hh_l0_reverse.data)

    print(f"  ✓ {layer_name} 權重轉換完成")

def convert_h5_to_pytorch(h5_path, output_path='converted_model.pth'):
    """轉換 h5 模型到 PyTorch"""
    print(f"\n{'='*60}")
    print(f"開始轉換: {h5_path}")
    print(f"輸出檔案: {output_path}")
    print(f"{'='*60}")

    # 載入 Keras 權重
    keras_weights = load_keras_weights(h5_path)

    # 建立 PyTorch 模型 (使用簡化版，匹配 Keras 架構)
    print("\n" + "="*60)
    print("建立 PyTorch 模型 (簡化版 - 匹配 Keras 架構)")
    print("="*60)

    pytorch_model = BiLSTMSimple(
        input_dim=22,
        hidden_dim=91,
        num_classes=4
    )

    # 轉換權重
    print("\n" + "="*60)
    print("開始權重轉換")
    print("="*60)

    # 轉換 LSTM1 權重
    convert_lstm_weights(keras_weights['lstm1'], pytorch_model.lstm1, 'LSTM1')

    # 轉換 LSTM2 權重
    convert_lstm_weights(keras_weights['lstm2'], pytorch_model.lstm2, 'LSTM2')

    # 轉換 Dense (輸出層) 權重
    print("\n轉換 Dense (輸出層) 權重...")
    dense_kernel = keras_weights['dense']['kernel']  # (182, 4)
    dense_bias = keras_weights['dense']['bias']  # (4,)

    # 現在模型架構匹配：BiLSTM(182) -> Dense(4)
    pytorch_model.fc.weight.data = torch.FloatTensor(dense_kernel.T)
    pytorch_model.fc.bias.data = torch.FloatTensor(dense_bias)
    print("  ✓ Dense 權重轉換完成")

    # 保存模型
    print("\n" + "="*60)
    print(f"保存模型到: {output_path}")
    print("="*60)

    torch.save(pytorch_model.state_dict(), output_path)

    print("\n✅ 轉換完成!")
    print(f"✅ 模型已保存到: {output_path}")

    # 驗證
    print("\n" + "="*60)
    print("驗證模型")
    print("="*60)

    pytorch_model.eval()
    test_input = torch.randn(1, 45, 22)  # batch=1, seq_len=45, features=22

    with torch.no_grad():
        output = pytorch_model(test_input)
        probs = torch.softmax(output, dim=1)

    print(f"\n測試輸入形狀: {test_input.shape}")
    print(f"模型輸出形狀: {output.shape}")
    print(f"輸出 logits: {output[0]}")
    print(f"輸出機率: {probs[0]}")
    print(f"預測類別: {torch.argmax(probs, dim=1).item()}")

    print("\n" + "="*60)
    print("⚠️  注意事項")
    print("="*60)
    print("1. 模型架構可能不完全匹配 (缺少 Attention, LayerNorm, fc1 等)")
    print("2. 建議檢查原始 Keras 模型的完整架構")
    print("3. 可能需要調整 PyTorch 模型以完全匹配 Keras 模型")
    print("4. 請用實際資料測試模型準確性")

    return pytorch_model

if __name__ == "__main__":
    h5_file = "final_forthesis_bidirectionallstm_and_encoders_exercise_classifier_model.h5"
    output_file = "backend/bilstm_mix_best_pt_converted.pth"

    if len(sys.argv) > 1:
        h5_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]

    model = convert_h5_to_pytorch(h5_file, output_file)
