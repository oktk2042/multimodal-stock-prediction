import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from arch.dlinear import DLinear
from arch.fusion_transformer import FusionTransformer
from arch.itransformer import iTransformer
from arch.lstm_attn import AttentionLSTM
from arch.patchtst import PatchTST
from arch.transformer import VanillaTransformer

warnings.filterwarnings("ignore")


# ==========================================
# 設定
# ==========================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
    INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
    OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production"

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 60
    PRED_LEN = 1

    # Text/Fin系カラムのキーワード (FusionTransformer用)
    TEXT_FIN_KEYWORDS = ["FinBERT", "Sentiment", "NetSales", "Operating", "Sales_to", "Log_NetSales"]

    # 期間設定（学習時と同じにする必要があります）
    TRAIN_END = "2023-12-31"
    VAL_END = "2024-12-31"


# ==========================================
# データセット定義 (学習コードと同一)
# ==========================================
class StockDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len - Config.PRED_LEN + 1

    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[
            idx + self.seq_len : idx + self.seq_len + Config.PRED_LEN
        ].squeeze(-1)


# ==========================================
# 関数群
# ==========================================
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            out = model(x)

            # モデル出力の形状調整
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out.squeeze(-1)

            # ターゲットとの形状合わせ
            if y.ndim == 1 and out.ndim == 2:
                out = out.squeeze(-1)
            elif y.ndim == 2 and out.ndim == 1:
                out = out.unsqueeze(-1)

            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def calculate_permutation_importance(model, dataset, feature_names):
    print("   -> Calculating Importance...")
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    # 重要度評価はMSEで統一
    criterion = nn.MSELoss()
    base_loss = validate(model, loader, criterion)

    importances = []
    original_X = dataset.X.clone()

    # 各特徴量を順番にシャッフルして影響を見る
    for i, feature in enumerate(tqdm(feature_names, desc="Permuting Features", leave=False)):
        permuted_X = original_X.clone()
        perm_idx = torch.randperm(permuted_X.size(0))
        permuted_X[:, i] = permuted_X[perm_idx, i]

        dataset.X = permuted_X
        perm_loader = DataLoader(dataset, batch_size=256, shuffle=False)
        perm_loss = validate(model, perm_loader, criterion)

        # Lossが増えた分だけ重要（正の値になるはず）
        imp = perm_loss - base_loss
        importances.append({"Feature": feature, "Importance": imp})

        # データを元に戻す
        dataset.X = original_X

    return pd.DataFrame(importances).sort_values("Importance", ascending=False)


def save_analysis_results(model_name, importance_df):
    # CSV保存
    csv_path = Config.OUTPUT_DIR / f"{model_name}_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # 上位20個の可視化
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()
    plt.savefig(Config.OUTPUT_DIR / f"{model_name}_feature_importance.png")
    plt.close()
    print(f"   -> Saved: {csv_path.name}")


# ==========================================
# メイン処理
# ==========================================
def main():
    print(f"Target Directory: {Config.OUTPUT_DIR}")

    # 1. データのロード
    print("Loading Data...")
    if not Config.INPUT_FILE.exists():
        print(f"Error: Input file not found: {Config.INPUT_FILE}")
        return

    df = pd.read_csv(Config.INPUT_FILE)
    exclude = ["Date", "code", "Name", "Sector", "Target_Return_5D", "Target_Close_5D", "Target_Return_1D", "Close"]
    feature_cols = [c for c in df.columns if c not in exclude]

    # カラムインデックス特定 (FusionTransformer用)
    text_cols_idx = []
    market_cols_idx = []
    for i, col in enumerate(feature_cols):
        is_text = any(keyword in col for keyword in Config.TEXT_FIN_KEYWORDS)
        if is_text:
            text_cols_idx.append(i)
        else:
            market_cols_idx.append(i)

    # 2. データセット作成 (Validationデータのみ)
    dates = pd.to_datetime(df["Date"])
    train_mask = dates <= Config.TRAIN_END
    val_mask = (dates > Config.TRAIN_END) & (dates <= Config.VAL_END)

    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    # 学習時と同じくTrainデータでfit
    scaler.fit(df.loc[train_mask, feature_cols].values)

    # Valデータをtransform
    X_val = scaler.transform(df.loc[val_mask, feature_cols].values)
    y_val = df.loc[val_mask, "Target_Return_5D"].values.reshape(-1, 1)

    val_ds = StockDataset(X_val, y_val, Config.SEQ_LEN)
    input_dim = len(feature_cols)

    # 3. パラメータ情報のロード
    summary_path = Config.OUTPUT_DIR / "model_comparison_summary.csv"
    if not summary_path.exists():
        print("Error: model_comparison_summary.csv が見つかりません。")
        return

    summary_df = pd.read_csv(summary_path)

    # 4. 各モデルで計算実行
    for _, row in summary_df.iterrows():
        model_name = row["Model"]
        print(f"\nProcessing {model_name}...")

        model_path = Config.OUTPUT_DIR / f"best_model_{model_name}.pth"
        if not model_path.exists():
            print(f"   -> Skip (Model file not found: {model_path})")
            continue

        try:
            model = None

            # パラメータ取得ヘルパー (NaN対策)
            def get_param(name, default=None, dtype=int):
                val = row.get(name)
                if pd.isna(val):
                    return default
                return dtype(val)

            # --- モデル構築 (学習コードと引数を合わせる) ---
            if model_name == "LSTM":
                model = AttentionLSTM(
                    input_dim,
                    Config.SEQ_LEN,
                    Config.PRED_LEN,
                    hidden_dim=get_param("hidden_dim", 64),
                    num_layers=get_param("num_layers", 2),
                    dropout=row["dropout"],
                )
            elif model_name == "DLinear":
                model = DLinear(input_dim, Config.SEQ_LEN, Config.PRED_LEN, individual=bool(row["individual"]))
            elif model_name == "Transformer":
                model = VanillaTransformer(
                    input_dim,
                    Config.SEQ_LEN,
                    Config.PRED_LEN,
                    d_model=get_param("d_model", 64),
                    n_heads=get_param("nhead", 4),  # CSVの列名はnheadだが、引数はn_heads
                    num_layers=get_param("layers", 2),
                    dropout=row["dropout"],
                )
            elif model_name == "PatchTST":
                patch_len = get_param("patch_len", 16)
                model = PatchTST(
                    input_dim,
                    Config.SEQ_LEN,
                    Config.PRED_LEN,
                    patch_len=patch_len,
                    stride=patch_len // 2,
                    d_model=get_param("d_model", 64),
                    n_heads=get_param("nhead", 4),  # ここも修正済み
                    num_layers=get_param("layers", 2),
                    dropout=row["dropout"],
                )
            elif model_name == "iTransformer":
                model = iTransformer(
                    input_dim,
                    Config.SEQ_LEN,
                    Config.PRED_LEN,
                    d_model=get_param("d_model", 64),
                    n_heads=get_param("nhead", 4),  # ここも修正済み
                    num_layers=get_param("layers", 2),
                    dropout=row["dropout"],
                )
            elif model_name == "FusionTransformer":
                model = FusionTransformer(
                    input_dim,
                    Config.SEQ_LEN,
                    Config.PRED_LEN,
                    market_cols_idx=market_cols_idx,
                    text_cols_idx=text_cols_idx,
                    d_model=get_param("d_model", 64),
                    n_heads=get_param("nhead", 4),  # ここも修正済み
                    num_layers=get_param("layers", 2),
                    dropout=row["dropout"],
                )

            if model is None:
                print(f"   -> Unknown model type: {model_name}")
                continue

            # 重みロード
            model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
            model.to(Config.DEVICE)

            # 重要度計算
            imp_df = calculate_permutation_importance(model, val_ds, feature_cols)
            save_analysis_results(model_name, imp_df)

        except Exception as e:
            print(f"   -> Error: {e}")
            import traceback

            traceback.print_exc()

    print("\nAll Done!")


if __name__ == "__main__":
    main()
