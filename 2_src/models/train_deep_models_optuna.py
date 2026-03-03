import copy
import random
import sys
import time
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from arch.dlinear import DLinear
from arch.fusion_transformer import FusionTransformer
from arch.itransformer import iTransformer
from arch.lstm_attn import AttentionLSTM
from arch.patchtst import PatchTST

# 各自のモデルファイルをインポート
from arch.transformer import VanillaTransformer

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 日本語フォント設定
matplotlib.rcParams["font.family"] = "MS Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set(font="MS Gothic")


# ==============================================================================
#  実験設定クラス (ExperimentConfig)
# ==============================================================================
class ExperimentConfig:
    # --------------------------------------------------------------------------
    # 1. パス・環境設定
    # --------------------------------------------------------------------------
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
    INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
    OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_deep_strict"

    SEED = 42
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------------------------------------------------------------
    # 2. データセット設定
    # --------------------------------------------------------------------------
    SEQ_LEN = 60
    PRED_LEN = 1
    TRAIN_END = "2023-12-31"
    VAL_END = "2024-12-31"

    # --------------------------------------------------------------------------
    # 3. 学習プロセス設定
    # --------------------------------------------------------------------------

    # [修正] 50回は多すぎます。15回で十分傾向は掴めます。
    N_TRIALS = 30

    # [修正] 300回は長すぎます。100回あれば十分収束します。
    EPOCHS = 100

    # [修正] 30回も待つ必要はありません。5回更新がなければ次へ行きます。
    PATIENCE = 10

    # --------------------------------------------------------------------------
    # 4. カラム定義（モダリティ分割用）
    # --------------------------------------------------------------------------
    # 疎なデータ（ニュース・決算・財務）のカラム名部分一致リスト
    # 新しいデータセットのカラム名に対応するためキーワードを追加
    TEXT_FIN_KEYWORDS = ["FinBERT", "Sentiment", "NetSales", "Operating", "Sales_to", "Log_NetSales", "Fin_", "News_"]

    # --------------------------------------------------------------------------
    # 5. ハイパーパラメータ探索空間
    # --------------------------------------------------------------------------

    # バッチサイズを大きくすると学習が速くなります。
    # PatchTSTが重すぎる場合、GPUメモリが許せば 256 や 512 を入れると高速化します。
    BATCH_SIZE_OPTS = [64, 128, 256]

    # [学習率]:
    # より細かく最適解を探れるよう、下限を 1e-5 まで広げます。
    LR_RANGE = (1e-5, 1e-2)

    # [ドロップアウト率]:
    DROPOUT_RANGE = (0.1, 0.5)

    # --- モデル別設定 ---
    LSTM_PARAMS = {"hidden_dim": [64, 128, 256], "num_layers": (1, 4)}
    TRANSFORMER_PARAMS = {
        "d_model": [64, 128],  # 重すぎるなら 256 を外して [64, 128] にする
        "n_heads": [4, 8],
        "layers": (2, 6),
    }
    PATCH_PARAMS = {
        "patch_len": [8, 16],
    }
    DLINEAR_PARAMS = {"individual": [True, False]}

    # FusionTransformer用設定
    FUSION_PARAMS = {
        "d_model": [64, 128, 256],  # 分割するので偶数である必要あり
        "n_heads": [4, 8, 16],
        "layers": (2, 4),
    }


# ==============================================================================
# 初期化処理
# ==============================================================================
ExperimentConfig.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 再現性重視


set_seed(ExperimentConfig.SEED)
print(f"Using Device: {ExperimentConfig.DEVICE}")
print(f"Config Loaded: SeqLen={ExperimentConfig.SEQ_LEN}, Epochs={ExperimentConfig.EPOCHS}")


# ==========================================
# 2. 損失関数の定義 (モデル別最適化)
# ==========================================
class DirectionalMSELoss(nn.Module):
    def __init__(self, alpha=2.0):  # alphaを 1.0 -> 2.0 に強化
        super().__init__()
        self.mse = nn.MSELoss()
        self.alpha = alpha

    def forward(self, pred, target):
        loss_mse = self.mse(pred, target)
        # 予測と正解の符号が違う場合にペナルティ
        diff_sign = torch.relu(-1.0 * pred * target)
        loss_dir = torch.mean(diff_sign)
        return loss_mse + self.alpha * loss_dir


def get_criterion(model_name):
    """モデルごとの推奨損失関数"""
    if model_name == "LSTM":
        return nn.SmoothL1Loss()  # Huber Loss (ノイズ対策)
    elif model_name == "DLinear":
        return nn.MSELoss()  # 標準的
    elif model_name in ["Transformer", "FusionTransformer"]:
        return DirectionalMSELoss(alpha=2.0)  # 方向重視
    elif model_name in ["PatchTST", "iTransformer"]:
        return nn.L1Loss()  # MAE (SOTA論文準拠)
    else:
        return nn.MSELoss()  # MSE as default


# ==========================================
# 3. データセット & ユーティリティ
# ==========================================
class StockDataset(Dataset):
    def __init__(self, X, y, seq_len):
        # 最初に一括でGPUに送る
        self.X = torch.tensor(X, dtype=torch.float32).to(ExperimentConfig.DEVICE)
        self.y = torch.tensor(y, dtype=torch.float32).to(ExperimentConfig.DEVICE)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len - ExperimentConfig.PRED_LEN + 1

    def __getitem__(self, idx):
        # GPU上のテンソルをそのまま返す（高速）
        return self.X[idx : idx + self.seq_len], self.y[
            idx + self.seq_len : idx + self.seq_len + ExperimentConfig.PRED_LEN
        ].squeeze(-1)


def load_data():
    start_time = time.time()
    cfg = ExperimentConfig
    print("\n[Loading] データの読み込みを開始します...")

    if not cfg.INPUT_FILE.exists():
        print(f"[Error] Input file not found: {cfg.INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(cfg.INPUT_FILE, encoding="utf-8-sig")

    if "Code" in df.columns:
        df.rename(columns={"Code": "code"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    print("[Info] 5日後の点予測を行うため、ターゲットを再計算します。")

    # ターゲット計算: (5日後終値 - 当日終値) / 当日終値
    df["Target_Return_5D"] = df.groupby("code")["Close"].transform(lambda x: np.log(x.shift(-5) / x))

    target_col = "Target_Return_5D"
    actual_price_col = "Target_Close_5D"
    df[actual_price_col] = df.groupby("code")["Close"].shift(-5)
    price_col = "Close"

    # 除外カラム
    possible_excludes = [
        "Date",
        "code",
        "Name",
        "Sector",
        target_col,
        actual_price_col,
        "Target_Return",
        "Target_Return_1D",
        price_col,
    ]

    exclude = [c for c in possible_excludes if c in df.columns]
    feature_cols = [c for c in df.columns if c not in exclude]

    # 欠損除去
    df = df.dropna(subset=[target_col]).fillna(0)
    df = df.reset_index(drop=True)

    dates = df["Date"]
    train_idx = df.index[dates <= cfg.TRAIN_END].values
    val_idx = df.index[(dates > cfg.TRAIN_END) & (dates <= cfg.VAL_END)].values
    test_idx = df.index[dates > cfg.VAL_END].values

    # Scaling (Features)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_idx, feature_cols].values)
    feature_data = scaler.transform(df[feature_cols].values)

    # --- Target Scaling ---
    # リターン値が小さすぎて学習が進まないためスケーリングを行う。
    # ただし、DirectionalLossで符号（正負）を使いたいので、平均は引かない (with_mean=False)。
    print("[Preprocessing] ターゲット変数のスケーリングを実行します (with_mean=False)")
    target_scaler = StandardScaler(with_mean=False)

    # Trainデータのみでfit
    train_targets = df.loc[train_idx, target_col].values.reshape(-1, 1)
    target_scaler.fit(train_targets)

    # 全データを変換し、1次元配列にする
    target_data = target_scaler.transform(df[target_col].values.reshape(-1, 1)).flatten()

    # Meta data
    meta_cols = ["code", "Name", "Date", price_col]
    if actual_price_col in df.columns:
        meta_cols.append(actual_price_col)
    else:
        meta_cols.append(target_col)

    if "Sector" in df.columns:
        meta_cols.insert(2, "Sector")

    meta_test = df.loc[test_idx, meta_cols].copy()

    # モダリティ特定
    text_cols_idx = []
    market_cols_idx = []
    text_col_names = []

    for i, col in enumerate(feature_cols):
        is_text = any(keyword in col for keyword in cfg.TEXT_FIN_KEYWORDS)
        if is_text:
            text_cols_idx.append(i)
            text_col_names.append(col)
        else:
            market_cols_idx.append(i)

    print(f"\n[Modality Info] Market Cols: {len(market_cols_idx)}, Text/Fin Cols: {len(text_cols_idx)}")
    print(f"  -> Text Cols: {text_col_names}")

    elapsed = time.time() - start_time
    print(f"[Done] データ読み込み完了 ({elapsed:.2f}秒)")
    print(f"      Features: {len(feature_cols)}, Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return {
        "feature_data": feature_data,
        "target_data": target_data,  # スケーリング済み
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "input_dim": len(feature_cols),
        "meta_test": meta_test,
        "features": feature_cols,
        "market_cols_idx": market_cols_idx,
        "text_cols_idx": text_cols_idx,
        "target_col_name": target_col,
        "actual_price_col_name": actual_price_col,
        "target_scaler": target_scaler,
    }


def calculate_metrics_df(df_res):
    """
    評価指標を計算
    """
    # 価格データ
    y_true_price = df_res["Actual"].values
    y_pred_price = df_res["Pred"].values
    y_curr_price = df_res["Current"].values

    # モデルの予測値 (これは対数収益率になっている)
    pred_return = df_res["Pred_Return"].values
    # 実測リターン = (5日後株価 - 現在株価) / 現在株価
    actual_return = np.log(y_true_price / y_curr_price)

    # 1. 価格ベースの誤差指標
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae = mean_absolute_error(y_true_price, y_pred_price)

    # 2. 方向正解率 (Accuracy)
    diff_true = y_true_price - y_curr_price
    diff_pred = y_pred_price - y_curr_price
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0変化を除外した厳密な符号一致率にする場合もありますが、ここではシンプルに符号比較
        accuracy = accuracy_score(np.sign(diff_true), np.sign(diff_pred)) * 100

    # 3. 決定係数 (R2)
    # R2_Price: 株価そのもののR2 (トレンドに乗っていれば高くなりやすい。0.99など)
    r2_price = r2_score(y_true_price, y_pred_price)

    # R2_Return: リターンのR2 (真の予測能力指標。プラスなら優秀)
    r2_return = r2_score(actual_return, pred_return)

    # 4. MAPE (価格)
    mask = y_true_price != 0
    if np.sum(mask) > 0:
        mape = np.mean(np.abs((y_true_price[mask] - y_pred_price[mask]) / y_true_price[mask])) * 100
    else:
        mape = np.nan

    # 5. 相関係数 (価格)
    if len(y_true_price) > 1:
        corr = np.corrcoef(y_true_price, y_pred_price)[0, 1]
    else:
        corr = np.nan

    return pd.Series(
        {
            "RMSE": rmse,
            "MAE": mae,
            "Accuracy": accuracy,
            "R2_Price": r2_price,
            "R2_Return": r2_return,
            "MAPE": mape,
            "Corr": corr,
        }
    )


def plot_sample_predictions(df_res, model_name, n_samples=3):
    codes = df_res["code"].unique()
    if len(codes) == 0:
        return
    samples = np.random.choice(codes, min(n_samples, len(codes)), replace=False)

    # 入力日(Date) + 5日 = 予測対象日
    shift_days = 5

    # 全期間プロット
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 4 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, code in zip(axes, samples):
        data = df_res[df_res["code"] == code].sort_values("Date").copy()

        # X軸用に日付をシフト
        target_dates = data["Date"] + pd.Timedelta(days=shift_days)

        ax.plot(target_dates, data["Actual"], label="Actual", color="black", alpha=0.6)
        ax.plot(target_dates, data["Pred"], label="Prediction", color="red", linestyle="--", alpha=0.8)

        ax.set_title(f"{model_name}: {code} - {data['Name'].iloc[0]} (Shifted to Target Date)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ExperimentConfig.OUTPUT_DIR / f"{model_name}_pred_full.png")
    plt.close()

    # 直近100日ズームプロット
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 4 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, code in zip(axes, samples):
        data = df_res[df_res["code"] == code].sort_values("Date").iloc[-100:].copy()
        if len(data) == 0:
            continue

        # X軸用に日付をシフト
        target_dates = data["Date"] + pd.Timedelta(days=shift_days)

        ax.plot(target_dates, data["Actual"], label="Actual", color="black", marker=".", alpha=0.6)
        ax.plot(target_dates, data["Pred"], label="Prediction", color="red", linestyle="--", marker=".", alpha=0.8)

        ax.set_title(f"{model_name}: {code} - {data['Name'].iloc[0]} (Zoom Last 100)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(ExperimentConfig.OUTPUT_DIR / f"{model_name}_pred_zoom.png")
    plt.close()


def plot_scatter_predictions(df_res, model_name):
    """実測リターン vs 予測リターンの散布図"""
    actual_ret = (df_res["Actual"] - df_res["Current"]) / df_res["Current"]
    pred_ret = df_res["Pred_Return"]

    plt.figure(figsize=(6, 6))
    plt.scatter(actual_ret, pred_ret, alpha=0.3, s=10)

    # 基準線 (y=x, y=0, x=0)
    max_val = max(actual_ret.max(), pred_ret.max())
    min_val = min(actual_ret.min(), pred_ret.min())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.5, label="Ideal")
    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    plt.title(f"{model_name}: Return Scatter Plot")
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(ExperimentConfig.OUTPUT_DIR / f"{model_name}_scatter.png")
    plt.close()


# ==========================================
# 4. 学習・評価関数
# ==========================================
def train_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    total_loss = 0

    for x, y in loader:
        optimizer.zero_grad()

        # autocastで演算をFP16化
        with torch.amp.autocast("cuda"):
            out = model(x)

            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out.squeeze(-1)
            if y.ndim == 1 and out.ndim == 2:
                out = out.squeeze(-1)
            elif y.ndim == 2 and out.ndim == 1:
                out = out.unsqueeze(-1)

            loss = criterion(out, y)

        # Scalerを使ってバックプロパゲーション
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)

            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out.squeeze(-1)
            if y.ndim == 1 and out.ndim == 2:
                out = out.squeeze(-1)
            elif y.ndim == 2 and out.ndim == 1:
                out = out.unsqueeze(-1)

            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)


def calculate_permutation_importance(model, dataset, feature_names):
    print("\n--- Calculating Feature Importance ---")
    model.eval()

    # 評価用Loader
    loader = DataLoader(
        dataset, batch_size=512, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False
    )
    eval_criterion = nn.MSELoss()

    # ベースライン
    base_loss = 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x)
            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out.squeeze(-1)
            if y.ndim == 1 and out.ndim == 2:
                out = out.squeeze(-1)
            elif y.ndim == 2 and out.ndim == 1:
                out = out.unsqueeze(-1)
            base_loss += eval_criterion(out, y).item() * x.size(0)
    base_loss /= len(dataset)

    importances = []
    original_X = dataset.X.clone()

    # TQDMで進捗表示
    for i, feature in enumerate(tqdm(feature_names, desc="Permuting", leave=False)):
        permuted_X = original_X.clone()
        perm_idx = torch.randperm(permuted_X.size(0))
        permuted_X[:, i] = permuted_X[perm_idx, i]

        dataset.X = permuted_X  # 一時的に差し替え

        curr_loss = 0
        with torch.no_grad():
            for x, y in loader:
                out = model(x)
                if isinstance(out, tuple):
                    out = out[0]
                if out.ndim == 3:
                    out = out.squeeze(-1)
                if y.ndim == 1 and out.ndim == 2:
                    out = out.squeeze(-1)
                elif y.ndim == 2 and out.ndim == 1:
                    out = out.unsqueeze(-1)
                curr_loss += eval_criterion(out, y).item() * x.size(0)
        curr_loss /= len(dataset)

        dataset.X = original_X  # 戻す

        imp = curr_loss - base_loss
        importances.append({"Feature": feature, "Importance": imp})

    return pd.DataFrame(importances).sort_values("Importance", ascending=False)


def save_analysis_results(model_name, importance_df):
    importance_df.to_csv(ExperimentConfig.OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False)
    plt.figure(figsize=(10, 8))
    sns.barplot(x="Importance", y="Feature", data=importance_df.head(20))
    plt.title(f"{model_name} Feature Importance")
    plt.tight_layout()
    plt.savefig(ExperimentConfig.OUTPUT_DIR / f"{model_name}_feature_importance.png")
    plt.close()


# ==========================================
# 5. Optuna Objective
# ==========================================
def objective(trial, data, model_name):
    # trial_start = time.time()
    input_dim = data["input_dim"]
    cfg = ExperimentConfig

    # 共通ハイパーパラメータ
    lr = trial.suggest_float("lr", cfg.LR_RANGE[0], cfg.LR_RANGE[1], log=True)
    batch_size = trial.suggest_categorical("batch_size", cfg.BATCH_SIZE_OPTS)
    dropout = trial.suggest_float("dropout", cfg.DROPOUT_RANGE[0], cfg.DROPOUT_RANGE[1])

    model = None
    if model_name == "LSTM":
        hidden_dim = trial.suggest_categorical("hidden_dim", cfg.LSTM_PARAMS["hidden_dim"])
        num_layers = trial.suggest_int("num_layers", cfg.LSTM_PARAMS["num_layers"][0], cfg.LSTM_PARAMS["num_layers"][1])
        model = AttentionLSTM(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "DLinear":
        individual = trial.suggest_categorical("individual", cfg.DLINEAR_PARAMS["individual"])
        model = DLinear(input_dim=input_dim, seq_len=cfg.SEQ_LEN, pred_len=cfg.PRED_LEN, individual=individual)

    elif model_name == "Transformer":
        d_model = trial.suggest_categorical("d_model", cfg.TRANSFORMER_PARAMS["d_model"])
        nhead = trial.suggest_categorical("nhead", cfg.TRANSFORMER_PARAMS["n_heads"])
        num_layers = trial.suggest_int(
            "layers", cfg.TRANSFORMER_PARAMS["layers"][0], cfg.TRANSFORMER_PARAMS["layers"][1]
        )
        model = VanillaTransformer(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "PatchTST":
        d_model = trial.suggest_categorical("d_model", cfg.TRANSFORMER_PARAMS["d_model"])
        nhead = trial.suggest_categorical("nhead", cfg.TRANSFORMER_PARAMS["n_heads"])
        num_layers = trial.suggest_int(
            "layers", cfg.TRANSFORMER_PARAMS["layers"][0], cfg.TRANSFORMER_PARAMS["layers"][1]
        )
        patch_len = trial.suggest_categorical("patch_len", cfg.PATCH_PARAMS["patch_len"])
        stride = patch_len
        model = PatchTST(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "iTransformer":
        d_model = trial.suggest_categorical("d_model", cfg.TRANSFORMER_PARAMS["d_model"])
        nhead = trial.suggest_categorical("nhead", cfg.TRANSFORMER_PARAMS["n_heads"])
        num_layers = trial.suggest_int(
            "layers", cfg.TRANSFORMER_PARAMS["layers"][0], cfg.TRANSFORMER_PARAMS["layers"][1]
        )
        model = iTransformer(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    elif model_name == "FusionTransformer":
        # ガード処理
        if not data["text_cols_idx"]:
            # OptunaのPruning例外を投げるか、あるいはWarningを出してreturn float('inf')
            print("[Skip] FusionTransformer: Text columns not found.")
            return float("inf")

        d_model = trial.suggest_categorical("d_model", cfg.FUSION_PARAMS["d_model"])
        nhead = trial.suggest_categorical("nhead", cfg.FUSION_PARAMS["n_heads"])
        num_layers = trial.suggest_int("layers", cfg.FUSION_PARAMS["layers"][0], cfg.FUSION_PARAMS["layers"][1])
        model = FusionTransformer(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            market_cols_idx=data["market_cols_idx"],
            text_cols_idx=data["text_cols_idx"],
            d_model=d_model,
            n_heads=nhead,
            num_layers=num_layers,
            dropout=dropout,
        )

    model = model.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = get_criterion(model_name)
    scaler = GradScaler()

    # 引数は3つ (X, y, seq_len)
    train_ds = StockDataset(
        data["feature_data"][data["train_idx"]], data["target_data"][data["train_idx"]], cfg.SEQ_LEN
    )
    val_ds = StockDataset(data["feature_data"][data["val_idx"]], data["target_data"][data["val_idx"]], cfg.SEQ_LEN)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False, persistent_workers=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False, persistent_workers=False
    )

    best_val_loss = float("inf")

    with tqdm(range(cfg.EPOCHS), desc=f"Trial {trial.number}", leave=False) as pbar:
        for epoch in pbar:
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
            val_loss = validate(model, val_loader, criterion)

            pbar.set_postfix({"T": f"{train_loss:.4f}", "V": f"{val_loss:.4f}"})
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
            else:
                if epoch > cfg.PATIENCE // 2 and val_loss > best_val_loss * 1.1:
                    break

    return best_val_loss


# ==========================================
# 6. ベストモデル学習 & メイン処理
# ==========================================
def train_best_model(model_name, best_params, data):
    print(f"\n--- Training Best {model_name} ---")
    cfg = ExperimentConfig
    input_dim = data["input_dim"]
    p = best_params

    # モデル構築
    if model_name == "LSTM":
        model = AttentionLSTM(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            hidden_dim=p["hidden_dim"],
            num_layers=p["num_layers"],
            dropout=p["dropout"],
        )

    elif model_name == "DLinear":
        model = DLinear(input_dim=input_dim, seq_len=cfg.SEQ_LEN, pred_len=cfg.PRED_LEN, individual=p["individual"])

    elif model_name == "Transformer":
        model = VanillaTransformer(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            d_model=p["d_model"],
            n_heads=p["nhead"],
            num_layers=p["layers"],
            dropout=p["dropout"],
        )

    elif model_name == "PatchTST":
        stride = p["patch_len"] // 2
        model = PatchTST(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            patch_len=p["patch_len"],
            stride=stride,
            d_model=p["d_model"],
            n_heads=p["nhead"],
            num_layers=p["layers"],
            dropout=p["dropout"],
        )

    elif model_name == "iTransformer":
        model = iTransformer(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            d_model=p["d_model"],
            n_heads=p["nhead"],
            num_layers=p["layers"],
            dropout=p["dropout"],
        )

    elif model_name == "FusionTransformer":
        model = FusionTransformer(
            input_dim=input_dim,
            seq_len=cfg.SEQ_LEN,
            pred_len=cfg.PRED_LEN,
            market_cols_idx=data["market_cols_idx"],
            text_cols_idx=data["text_cols_idx"],
            d_model=p["d_model"],
            n_heads=p["nhead"],
            num_layers=p["layers"],
            dropout=p["dropout"],
        )

    model = model.to(cfg.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=p["lr"])
    criterion = get_criterion(model_name)
    scaler = GradScaler()

    # データセット
    train_ds = StockDataset(
        data["feature_data"][data["train_idx"]], data["target_data"][data["train_idx"]], cfg.SEQ_LEN
    )
    val_ds = StockDataset(data["feature_data"][data["val_idx"]], data["target_data"][data["val_idx"]], cfg.SEQ_LEN)
    test_ds = StockDataset(data["feature_data"][data["test_idx"]], data["target_data"][data["test_idx"]], cfg.SEQ_LEN)

    bs = p["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False)

    best_loss = float("inf")
    best_weights = copy.deepcopy(model.state_dict())
    patience_cnt = 0
    train_hist, val_hist = [], []

    pbar = tqdm(range(cfg.EPOCHS), desc=f"Best {model_name}")
    for epoch in pbar:
        t_loss = train_epoch(model, train_loader, optimizer, criterion, scaler)
        v_loss = validate(model, val_loader, criterion)
        train_hist.append(t_loss)
        val_hist.append(v_loss)
        pbar.set_postfix({"V": f"{v_loss:.5f}"})

        if v_loss < best_loss:
            best_loss = v_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.PATIENCE:
                print(f" Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_weights)
    torch.save(model.state_dict(), cfg.OUTPUT_DIR / f"best_model_{model_name}.pth")

    # --- 推論 & 逆変換 ---
    model.eval()
    preds_scaled = []
    gate_scores = []

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(cfg.DEVICE)
            # FusionTransformerの場合のみGate取得
            if model_name == "FusionTransformer":
                out, gate = model(x, return_gate=True)
                if gate is not None:
                    # [Batch, Seq, Dim] -> [Batch] (直近時刻平均)
                    gate_scores.extend(gate[:, -1, :].mean(dim=1).cpu().numpy())
                else:
                    gate_scores.extend([0.0] * x.size(0))
            else:
                out = model(x)
                gate_scores.extend([0.0] * x.size(0))

            if isinstance(out, tuple):
                out = out[0]
            if out.ndim == 3:
                out = out.squeeze(-1)
            preds_scaled.extend(out.cpu().numpy().flatten())

    # 予測値を元のリターンのスケールに戻す (Inverse Transform)
    preds_scaled = np.array(preds_scaled).reshape(-1, 1)
    preds_original = data["target_scaler"].inverse_transform(preds_scaled).flatten()

    # 結果保存
    res_df = data["meta_test"].iloc[cfg.SEQ_LEN :].reset_index(drop=True)
    min_len = min(len(res_df), len(preds_original))
    res_df = res_df.iloc[:min_len].copy()

    # 逆変換した値を入れる
    res_df["Pred_Return"] = preds_original[:min_len]

    # 対数収益率からの復元
    res_df["Pred"] = res_df["Close"] * np.exp(res_df["Pred_Return"])

    # 実測値の復元
    if data["actual_price_col_name"] in res_df.columns:
        res_df["Actual"] = res_df[data["actual_price_col_name"]]
    else:
        pass

    res_df["Current"] = res_df["Close"]
    if model_name == "FusionTransformer":
        res_df["Gate_Score"] = gate_scores[:min_len]

    res_df.to_csv(cfg.OUTPUT_DIR / f"predictions_{model_name}.csv", index=False)
    metrics = calculate_metrics_df(res_df)

    return metrics, train_hist, val_hist, res_df, model, val_ds


def main():
    # 全体の開始時間
    total_start = time.time()
    cfg = ExperimentConfig

    print("\n" + "=" * 60)
    print(f" プロジェクト開始: {cfg.PROJECT_ROOT}")
    print(f" デバイス: {cfg.DEVICE}")
    print("=" * 60)

    data = load_data()

    # 集計用リスト
    summary = []
    # 実験対象の全モデルリスト（論文比較用フルセット）
    models_to_run = [
        # "LSTM",               # 古典的ベースライン
        # "DLinear",            # シンプルで強力なベースライン
        # "Transformer",        # Vanilla Transformer
        # "PatchTST",           # SOTAモデル (時系列特化)
        # "iTransformer",       # SOTAモデル (変数の反転)
        # "FusionTransformer"   # 提案手法 (ニュース融合)
    ]

    print(f"\n[Plan] 以下の {len(models_to_run)} モデルの最適化を実行します:")
    print(f"       {', '.join(models_to_run)}")

    for i, model_name in enumerate(models_to_run):
        print("\n" + "-" * 60)
        print(f" [{i + 1}/{len(models_to_run)}] {model_name} の最適化を開始...")
        print("-" * 60)

        model_start = time.time()

        study = optuna.create_study(direction="minimize")
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        try:
            # 探索回数の調整
            if model_name in ["PatchTST", "iTransformer"]:
                n_trials = 20  # 重いモデルは少なめ
            else:
                n_trials = 30  # 軽いモデル(DLinear等)は多め

            # 1. Optuna探索
            study.optimize(lambda trial: objective(trial, data, model_name), n_trials=n_trials)

            # 2. ベストモデルで再学習 & 評価
            metrics, t_hist, v_hist, res_df, model, val_ds = train_best_model(model_name, study.best_params, data)

            # 3. 結果の集約
            res_dict = {"Model": model_name}
            res_dict.update(metrics.to_dict())
            res_dict.update(study.best_params)
            summary.append(res_dict)

            # 4. 特徴量重要度計算
            # DLinearなどは特徴量重要度が出しにくい(または計算が重い)が、一律計算してみる
            imp_df = calculate_permutation_importance(model, val_ds, data["features"])
            save_analysis_results(model_name, imp_df)

            # 5. 予測結果のプロット保存
            plot_sample_predictions(res_df, model_name)
            plot_scatter_predictions(res_df, model_name)

            # 6. 学習曲線の保存
            plt.figure(figsize=(10, 5))
            plt.plot(t_hist, label="Train Loss")
            plt.plot(v_hist, label="Val Loss")
            plt.title(f"{model_name} Learning Curve")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(cfg.OUTPUT_DIR / f"{model_name}_learning_curve.png")
            plt.close()

            print(f"[Completed] {model_name} 全工程完了 ({(time.time() - model_start) / 60:.1f}分)")

        except KeyboardInterrupt:
            print("\n[Interrupt] ユーザーによる中断。")
            break
        except Exception as e:
            print(f"\n[Error] {model_name} でエラーが発生しました: {e}")
            import traceback

            traceback.print_exc()

    # --- 過去の計算結果（LSTM, Transformer）を統合 ---
    print("\n" + "=" * 60)
    print(" 既存の結果ファイル(predictions_*.csv)を統合します...")
    print("=" * 60)

    # 出力フォルダ内のすべての予測ファイルを探す
    pred_files = list(cfg.OUTPUT_DIR.glob("predictions_*.csv"))

    for p_file in pred_files:
        # ファイル名からモデル名を抽出 (predictions_LSTM.csv -> LSTM)
        m_name = p_file.stem.replace("predictions_", "")

        # 今回の実行リストに含まれていない（＝過去にやった）モデルなら読み込む
        if m_name not in models_to_run:
            # 既にsummaryに入っていないか確認（重複防止）
            if any(d["Model"] == m_name for d in summary):
                continue

            print(f" -> 過去の計算結果を復元中: {m_name}")
            try:
                df_past = pd.read_csv(p_file)
                # 指標を再計算
                metrics_past = calculate_metrics_df(df_past)

                res_dict = {"Model": m_name}
                res_dict.update(metrics_past.to_dict())
                res_dict["Note"] = "Existing Result"  # 目印

                summary.append(res_dict)
            except Exception as e:
                print(f"    [Warning] 読み込み失敗: {e}")

    # --- 最終結果の表示と保存 ---
    total_time = time.time() - total_start
    print("\n" + "=" * 60)
    print(f" 全工程完了 (所要時間: {total_time / 60:.1f}分)")
    print("=" * 60)

    if summary:
        # R2_Return (リターンの予測精度) が高い順にソート
        summary_df = pd.DataFrame(summary).sort_values("R2_Return", ascending=False)

        # 保存
        csv_path = cfg.OUTPUT_DIR / "model_comparison_summary.csv"
        summary_df.to_csv(csv_path, index=False)
        print(f"\n[Output] 比較結果を保存しました: {csv_path}")

        # 表示項目の選択
        disp_cols = ["Model", "R2_Return", "Accuracy", "R2_Price", "RMSE", "MAE", "Corr"]
        print("\n【最終結果ランキング (評価指標: R2_Return)】")
        print(summary_df[disp_cols].to_string(index=False))

        # 全モデル比較グラフ作成 (Accuracy)
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Model", y="Accuracy", data=summary_df, palette="viridis")
        plt.title("Model Accuracy Comparison")
        plt.ylim(40, 60)
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(cfg.OUTPUT_DIR / "model_accuracy_comparison.png")
        plt.close()

        # R2_Returnの比較グラフ
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Model", y="R2_Return", data=summary_df, palette="magma")
        plt.title("Model Return R2 Score Comparison (Higher is Better)")
        plt.grid(axis="y", alpha=0.3)
        plt.axhline(0, color="black", linewidth=0.8)
        plt.savefig(cfg.OUTPUT_DIR / "model_return_r2_comparison.png")
        plt.close()

    else:
        print("[Warn] 結果が得られませんでした。")


if __name__ == "__main__":
    main()
