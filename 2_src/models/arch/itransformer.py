import torch.nn as nn

from layers.revin import RevIN
from layers.self_attention_family import Encoder, EncoderLayer


class iTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=512, n_heads=8, num_layers=2, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        self.revin = RevIN(input_dim)
        self.encoder_embedding = nn.Linear(seq_len, d_model)

        self.encoder = Encoder(
            [
                EncoderLayer(d_model, n_heads, d_ff=d_model * 4, dropout=dropout, norm_first=True)
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        self.projection = nn.Linear(d_model, pred_len)

        # 各変数の予測結果(M個)を1つのターゲット(Return)に統合する層
        self.output_mix = nn.Linear(input_dim * pred_len, pred_len)

    def forward(self, x):
        x = self.revin(x, "norm")

        # [Batch, Seq, Chan] -> [Batch, Chan, Seq]
        x = x.permute(0, 2, 1)

        x = self.encoder_embedding(x)
        x, attns = self.encoder(x)
        x = self.projection(x)  # [Batch, Chan, Pred_Len]

        x = x.reshape(x.shape[0], -1)  # [Batch, Chan * Pred_Len]
        x = self.output_mix(x)  # [Batch, Pred_Len]

        return x
