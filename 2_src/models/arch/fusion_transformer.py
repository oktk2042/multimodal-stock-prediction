import torch
import torch.nn as nn

from layers.embed import PositionalEmbedding
from layers.revin import RevIN
from layers.self_attention_family import Encoder, EncoderLayer


class FusionTransformer(nn.Module):
    """
    FusionTransformer (Final Rigorous Version)
    - Architecture: Encoder-Only Transformer with Multi-Modal Fusion
    - Key Components: RevIN, 1D-CNN, Gated Cross-Attention, Positional Embedding
    """

    def __init__(
        self,
        input_dim,
        seq_len,
        pred_len,
        market_cols_idx,
        text_cols_idx,
        d_model=128,
        n_heads=4,
        num_layers=2,
        dropout=0.1,
        ablation_mode="none",
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.ablation_mode = ablation_mode

        # GPUエラー対策: リストをTensorに変換しておく
        self.market_idx = torch.tensor(market_cols_idx, dtype=torch.long)
        self.text_idx = torch.tensor(text_cols_idx, dtype=torch.long)

        dim_market = len(market_cols_idx)
        dim_text = len(text_cols_idx)

        # 1. 【RevIN】非定常性（分布シフト）への対策
        self.revin = RevIN(input_dim)

        # 2. 【Feature Extraction】
        # Market Path: 1D-CNNで局所的なトレンドを抽出
        if self.ablation_mode == "no_cnn":
            self.market_encoder = nn.Linear(dim_market, d_model)
        else:
            self.market_encoder = nn.Sequential(
                nn.Conv1d(in_channels=dim_market, out_channels=d_model, kernel_size=3, padding=1),
                nn.BatchNorm1d(d_model),
                nn.GELU(),  # GELU活性化関数
                nn.Dropout(dropout),
            )

        # Text Path: 次元圧縮
        self.text_proj = nn.Linear(dim_text, d_model)

        # 3. 【Positional Embedding】時系列順序の注入（必須要素）
        self.pos_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

        # 4. 【Fusion Mechanism】Gated Cross-Attention
        # Cross-Attention: 数値データ(Query)に関連するテキスト(Key/Value)を抽出
        self.cross_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True, dropout=dropout)

        # Gating Mechanism: 抽出した情報がどれくらい有用かを判断し、通過量を0~1で制御
        # これにより「ニュースがない日」や「無関係なニュース」を遮断する
        self.gate_linear = nn.Linear(d_model, d_model)
        self.norm_fusion = nn.LayerNorm(d_model)

        # 5. 【Transformer Encoder】長期依存関係の学習
        self.encoder = Encoder(
            [
                EncoderLayer(d_model, n_heads, d_ff=d_model * 4, dropout=dropout, norm_first=True)
                for _ in range(num_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )

        # 6. 【Prediction Head】予測層
        # 最終時刻の特徴量のみを使用する設計
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, return_gate=False):
        # x: [Batch, Seq_Len, Input_Dim]

        # --- 1. RevIN (Normalization) ---
        # 入力の分布シフト対策としてNormは実施する
        x = self.revin(x, "norm")

        # Modality Split
        if self.market_idx.device != x.device:
            self.market_idx = self.market_idx.to(x.device)
            self.text_idx = self.text_idx.to(x.device)

        x_market = x[:, :, self.market_idx]
        x_text = x[:, :, self.text_idx]

        if self.ablation_mode == "no_text":
            x_text = torch.zeros_like(x_text)

        # --- 2. Feature Extraction ---
        # Market: CNN [Batch, Seq, d_model]
        if self.ablation_mode != "no_cnn":
            x_m = x_market.permute(0, 2, 1)
            x_m = self.market_encoder(x_m)
            x_m = x_m.permute(0, 2, 1)
        else:
            x_m = self.market_encoder(x_market)

        # Text: Linear [Batch, Seq, d_model]
        x_t = self.text_proj(x_text)

        # --- 3. Add Positional Embedding ---
        pe = self.pos_embedding(x_m)
        x_m = x_m + pe
        x_t = x_t + pe

        x_m = self.dropout(x_m)
        x_t = self.dropout(x_t)

        # --- 4. Fusion (Gated Cross-Attention) ---
        gate_value = None  # 保存用変数

        if self.ablation_mode == "no_gate":
            z = x_m
        else:
            # Query=Market, Key/Value=Text
            attn_out, _ = self.cross_attn(query=x_m, key=x_t, value=x_t)

            # Gating: Attention出力をさらにSigmoidで重み付け
            gate_score = self.gate_linear(attn_out)
            gate = torch.sigmoid(gate_score)

            # Gateの値を保存（分析用）
            # [Batch, Seq, d_model] なので、直近時刻の平均値などを取るのが一般的
            if return_gate:
                gate_value = gate

            fused_info = gate * attn_out

            # 残差結合
            z = self.norm_fusion(x_m + fused_info)

        # --- 5. Transformer Encoder ---
        z, _ = self.encoder(z)

        # --- 6. Prediction ---
        # 多くの時系列SOTAと同様、シーケンスの最後の情報を利用
        out = self.head(z[:, -1, :])

        # RevIN (Denormalization) を削除
        # 今回の予測ターゲット(Return)は入力(Price等)とは異なる分布のため、
        # 入力の統計情報で復元してはいけません。
        # out = self.revin(out, 'denorm')  <-- これがバグの原因でした

        # return_gateがTrueなら、予測値とGate値を返す
        if return_gate:
            return out, gate_value

        return out
