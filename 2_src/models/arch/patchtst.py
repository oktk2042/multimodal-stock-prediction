import torch
import torch.nn as nn

from layers.revin import RevIN
from layers.self_attention_family import Encoder, EncoderLayer


class PatchTST(nn.Module):
    def __init__(
        self, input_dim, seq_len, pred_len, patch_len=16, stride=8, d_model=128, n_heads=4, num_layers=3, dropout=0.1
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.input_dim = input_dim

        # forward時のパディングロジックと厳密に合わせる
        pad_len = patch_len - stride
        if (seq_len - patch_len) % stride != 0:
            pad_len += stride - ((seq_len - patch_len) % stride)
        self.pad_len = pad_len
        seq_len_padded = seq_len + pad_len
        self.num_patches = (seq_len_padded - patch_len) // stride + 1

        self.revin = RevIN(input_dim, affine=True)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(d_model, n_heads, d_ff=d_model * 4, dropout=dropout, norm_first=True)
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # 全変数(input_dim) × 全パッチ(num_patches) × 特徴量(d_model) を入力とする
        self.head = nn.Linear(input_dim * self.num_patches * d_model, pred_len)

    def forward(self, x, mask_ratio=0.0):
        # x: [Batch, Seq, Chan]
        x = self.revin(x, "norm")
        B, L, M = x.shape

        x = x.permute(0, 2, 1).reshape(B * M, L, 1)
        x = x.permute(0, 2, 1)
        x = nn.functional.pad(x, (0, self.pad_len), mode="replicate")

        x = x.unfold(dimension=2, size=self.patch_len, step=self.stride)
        x = x.reshape(B * M, -1, self.patch_len)

        x = self.patch_embedding(x) + self.position_embedding  # Broadcast
        x = self.dropout(x)
        x, attns = self.encoder(x)  # [B*M, Num_Patches, d_model]

        # [B*M, N, D] -> [B, M, N*D] -> [B, M*N*D] -> Linear
        x = x.reshape(B, M, -1)
        x = x.reshape(B, -1)
        x = self.head(x)  # [B, Pred_Len]

        # RevINのdenormは、出力が「入力変数の復元」ではないため(Target予測なので)行いません
        return x
