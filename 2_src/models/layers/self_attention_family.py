import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullAttention(nn.Module):
    """
    論文準拠の厳密なAttention計算。
    Q, K, V の射影から Attention Score 計算、Softmax までを明示的に記述。
    """

    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.dropout = nn.Dropout(dropout)
        self.output_attention = output_attention

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)

    def forward(self, x_q, x_k, x_v, attn_mask=None):
        B, L, _ = x_q.shape
        _, S, _ = x_k.shape
        H = self.n_heads

        # 1. 射影 & ヘッド分割: [B, L, D] -> [B, L, H, d_k] -> [B, H, L, d_k]
        Q = self.W_Q(x_q).view(B, L, H, self.d_k).transpose(1, 2)
        K = self.W_K(x_k).view(B, S, H, self.d_k).transpose(1, 2)
        V = self.W_V(x_v).view(B, S, H, self.d_v).transpose(1, 2)

        # 2. Scaled Dot-Product Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # [B, H, L, S]

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        attn_weights = torch.softmax(scores, dim=-1)

        # 3. Dropout & Context
        A = self.dropout(attn_weights)
        V_out = torch.matmul(A, V)  # [B, H, L, d_v]

        # 4. 結合: [B, H, L, d_v] -> [B, L, H, d_v] -> [B, L, D]
        V_out = V_out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        output = self.out_projection(V_out)

        if self.output_attention:
            return output, attn_weights
        else:
            return output, None


class EncoderLayer(nn.Module):
    """
    Pre-Norm (iTransformer/PatchTST推奨) と Post-Norm (Vanilla Transformer) を選択可能にした厳密なレイヤー
    """

    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, activation="gelu", norm_first=False):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.norm_first = norm_first  # TrueならPre-Norm (最近のトレンド), FalseならPost-Norm

        self.attention = FullAttention(d_model, n_heads, dropout=dropout, output_attention=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Feed Forward
        self.conv1 = nn.Linear(d_model, d_ff)
        self.conv2 = nn.Linear(d_ff, d_model)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, attn_mask=None):
        # Pre-Norm: Norm -> Attn -> Add
        if self.norm_first:
            x_norm = self.norm1(x)
            new_x, attn = self.attention(x_norm, x_norm, x_norm, attn_mask=attn_mask)
            x = x + self.dropout(new_x)

            x_norm = self.norm2(x)
            ff_out = self.conv2(self.dropout(self.activation(self.conv1(x_norm))))
            x = x + self.dropout(ff_out)

        # Post-Norm: Attn -> Add -> Norm (Original Transformer)
        else:
            new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
            x = x + self.dropout(new_x)
            x = self.norm1(x)

            ff_out = self.conv2(self.dropout(self.activation(self.conv1(x))))
            x = x + self.dropout(ff_out)
            x = self.norm2(x)

        return x, attn


class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for layer in self.layers:
            x, attn = layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
