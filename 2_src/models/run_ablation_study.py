import copy
import random
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from arch.fusion_transformer import FusionTransformer

# パス設定 (arch/layersを読み込むため)
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))


# 再現性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        diff_sign = torch.relu(-1.0 * pred * target)
        loss_dir = torch.mean(diff_sign)
        return loss_mse + self.alpha * loss_dir


warnings.filterwarnings("ignore")


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
    INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
    TARGET_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
    OUTPUT_DIR = TARGET_DIR
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEQ_LEN = 60
    PRED_LEN = 1
    EPOCHS = 50  # アブレーション用なので少し減らしても良い
    PATIENCE = 10
    TEXT_FIN_KEYWORDS = ["FinBERT", "Sentiment", "NetSales", "Operating", "Sales_to", "Log_NetSales", "Yboard"]
    TRAIN_END = "2023-12-31"
    VAL_END = "2024-12-31"


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


def load_data_and_params():
    print("Loading Data & Params...")
    df = pd.read_csv(Config.INPUT_FILE, dtype={"Code": str, "code": str})
    if "Code" in df.columns:
        df.rename(columns={"Code": "code"}, inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    print("Generating Target_Return_5D...")

    # 5日後の終値との変化率: (Close_5d_after - Close) / Close
    df["Target_Return_5D"] = df.groupby("code")["Close"].transform(lambda x: (x.shift(-5) - x) / x)
    df["Target_Close_5D"] = df.groupby("code")["Close"].shift(-5)

    # ターゲット欠損を除去
    df = df.dropna(subset=["Target_Return_5D"]).fillna(0)
    df = df.reset_index(drop=True)

    # 特徴量カラムの選定
    exclude = [
        "Date",
        "code",
        "Name",
        "Sector",
        "Target_Return_5D",
        "Target_Close_5D",
        "Target_Return",
        "Target_Return_1D",
        "Close",
    ]

    # 数値型のみ抽出
    feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

    # パラメータ読み込み (なければデフォルト)
    best_params = {"d_model": 64, "n_heads": 4, "num_layers": 2, "lr": 0.001, "batch_size": 128, "dropout": 0.1}

    # カラムインデックス特定
    text_cols_idx = []
    market_cols_idx = []
    for i, col in enumerate(feature_cols):
        if any(k in col for k in Config.TEXT_FIN_KEYWORDS):
            text_cols_idx.append(i)
        else:
            market_cols_idx.append(i)

    return df, feature_cols, market_cols_idx, text_cols_idx, best_params


def train_variant(mode_name, ablation_mode, data_pack):
    print(f"\n--- Training Variant: {mode_name} (Mode: {ablation_mode}) ---")
    set_seed(42)
    df, features, m_idx, t_idx, params = data_pack

    # Split
    dates = df["Date"]
    train_mask = dates <= Config.TRAIN_END
    val_mask = (dates > Config.TRAIN_END) & (dates <= Config.VAL_END)
    test_mask = dates > Config.VAL_END

    # Scaling
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_raw = df.loc[train_mask, features].values
    scaler.fit(X_train_raw)

    X_train = scaler.transform(df.loc[train_mask, features].values)
    y_train = df.loc[train_mask, "Target_Return_5D"].values
    X_val = scaler.transform(df.loc[val_mask, features].values)
    y_val = df.loc[val_mask, "Target_Return_5D"].values
    X_test = scaler.transform(df.loc[test_mask, features].values)
    y_test = df.loc[test_mask, "Target_Return_5D"].values

    train_ds = StockDataset(X_train, y_train, Config.SEQ_LEN)
    val_ds = StockDataset(X_val, y_val, Config.SEQ_LEN)
    test_ds = StockDataset(X_test, y_test, Config.SEQ_LEN)

    bs = int(params["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    model = FusionTransformer(
        input_dim=len(features),
        seq_len=Config.SEQ_LEN,
        pred_len=Config.PRED_LEN,
        market_cols_idx=m_idx,
        text_cols_idx=t_idx,
        d_model=int(params["d_model"]),
        n_heads=int(params["n_heads"]),
        num_layers=int(params["num_layers"]),
        dropout=params["dropout"],
        ablation_mode=ablation_mode,
    ).to(Config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    criterion = DirectionalMSELoss(alpha=1.0)

    best_loss = float("inf")
    patience = 0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(Config.EPOCHS):
        model.train()
        for x, y in train_loader:
            x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
            optimizer.zero_grad()
            out = model(x).squeeze(-1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(Config.DEVICE), y.to(Config.DEVICE)
                out = model(x).squeeze(-1)
                val_loss += criterion(out, y).item()
        val_loss /= len(val_loader)

        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= Config.PATIENCE:
                break

    model.load_state_dict(best_weights)
    model.eval()
    preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(Config.DEVICE)
            out = model(x).squeeze(-1)
            preds.extend(out.cpu().numpy())

    # Accuracy Check
    test_df = df.loc[test_mask].iloc[Config.SEQ_LEN :].reset_index(drop=True)
    min_len = min(len(test_df), len(preds))
    test_df = test_df.iloc[:min_len]
    preds = np.array(preds[:min_len])

    diff_true = test_df["Target_Close_5D"] - test_df["Close"]
    diff_pred = (test_df["Close"] * (1 + preds)) - test_df["Close"]

    accuracy = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100

    # News Impact
    # FinBERTスコアのカラム名チェック
    score_col = "Fin_FinBERT_Score" if "Fin_FinBERT_Score" in test_df.columns else "FinBERT_Score"
    if score_col in test_df.columns:
        mask_news = test_df[score_col] != 0
        acc_news = np.mean(np.sign(diff_true[mask_news]) == np.sign(diff_pred[mask_news])) * 100
        acc_no = np.mean(np.sign(diff_true[~mask_news]) == np.sign(diff_pred[~mask_news])) * 100
    else:
        acc_news, acc_no = 0, 0

    return {"Variant": mode_name, "Accuracy": accuracy, "Acc_With_News": acc_news, "Acc_No_News": acc_no}


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_pack = load_data_and_params()

    variants = [("Proposed (Full)", "none"), ("w/o Text", "no_text"), ("w/o Gating", "no_gate"), ("w/o CNN", "no_cnn")]

    results = []
    for name, mode in variants:
        results.append(train_variant(name, mode, data_pack))

    df_res = pd.DataFrame(results)
    print("\n", df_res)
    df_res.to_csv(Config.OUTPUT_DIR / "ablation_study_results.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.bar(df_res["Variant"], df_res["Accuracy"], color="skyblue")
    plt.title("Ablation Study Accuracy")
    plt.ylim(40, 60)
    plt.savefig(Config.OUTPUT_DIR / "ablation_study_chart.png")


if __name__ == "__main__":
    main()
