import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLSTM(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

        # Score = v^T * tanh(W_h * encoder_outputs + W_q * query)
        self.attn_W = nn.Linear(hidden_dim, hidden_dim)  # Encoder Outputs用
        self.attn_U = nn.Linear(hidden_dim, hidden_dim)  # Query (Final Hidden)用 <--- 追加
        self.attn_v = nn.Linear(hidden_dim, 1, bias=False)
        self.decoder = nn.Linear(hidden_dim, pred_len)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Query: 最終層のHidden State [B, 1, H]
        query = h_n[-1].unsqueeze(1)

        # Keys: 全時刻のHidden State [B, Seq, H]
        keys = lstm_out

        # Additive Attention: tanh(W*keys + U*query)
        # queryをブロードキャストして加算します
        energy = torch.tanh(self.attn_W(keys) + self.attn_U(query))

        attention_scores = self.attn_v(energy).squeeze(2)  # [B, Seq]
        alpha = F.softmax(attention_scores, dim=1).unsqueeze(1)  # [B, 1, Seq]

        context = torch.bmm(alpha, lstm_out).squeeze(1)  # [B, Hidden]

        out = self.decoder(context)
        return out
