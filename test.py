import torch
import torch.nn as nn

# Attention 模組
class Attention(nn.Module):
    def __init__(self, hidden_dim, attn_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim * 2, attn_dim)
        self.fc2 = nn.Linear(attn_dim, 1)

    def forward(self, x):
        # x: [batch, seq_len, hidden_dim*2]
        weights = torch.softmax(self.fc2(torch.relu(self.fc1(x))), dim=1)  # [batch, seq_len, 1]
        context = torch.sum(weights * x, dim=1)  # [batch, hidden_dim*2]
        return context

# BiLSTM + Attention 模型
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

if __name__ == "__main__":
    # 載入權重
    state_dict = torch.load("bilstm_mix_best_pt.pth", map_location="cpu")
    model = BiLSTMAttention(input_dim=102, hidden_dim=96, attn_dim=128, num_classes=5)
    model.load_state_dict(state_dict)
    model.eval()

    print("模型載入成功！")
    print(model)

    # 測試一筆假資料
    x = torch.randn(1, 10, 102)  # batch=1, seq_len=10, input_dim=102
    y = model(x)
    print("輸出 shape:", y.shape)
    print("輸出結果:", y)

