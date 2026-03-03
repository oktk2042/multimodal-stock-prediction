import torch.nn as nn

from layers.embed import DataEmbedding
from layers.revin import RevIN
from layers.self_attention_family import Encoder, EncoderLayer


class VanillaTransformer(nn.Module):
    def __init__(self, input_dim, seq_len, pred_len, d_model=64, n_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.revin = RevIN(input_dim)
        self.embedding = DataEmbedding(input_dim, d_model, dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(d_model, n_heads, d_ff=d_model * 4, dropout=dropout, norm_first=False)
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # 最後の1点だけでなく、全シーケンス(seq_len)を平坦化して入力します
        self.output_linear = nn.Linear(seq_len * d_model, pred_len)

    def forward(self, x):
        x = self.revin(x, "norm")
        x = self.embedding(x)
        x, attns = self.encoder(x)

        # x: [Batch, Seq_Len, d_model] -> [Batch, Seq_Len * d_model]
        x = x.reshape(x.shape[0], -1)
        x = self.output_linear(x)  # [Batch, pred_len]

        return x
