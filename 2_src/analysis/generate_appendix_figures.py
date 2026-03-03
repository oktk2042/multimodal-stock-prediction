from pathlib import Path

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 設定 (Windows用日本語フォント)
# ==========================================
matplotlib.rcParams["font.family"] = "MS Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set(font="MS Gothic", style="whitegrid")

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production"  # CSVがある場所
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "final_figures"  # 画像出力先

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 対象モデル全リスト
ALL_MODELS = ["LSTM", "DLinear", "Transformer", "PatchTST", "iTransformer", "FusionTransformer"]

print(f"画像保存先: {OUTPUT_DIR}")

# ==========================================
# 2. プロット関数群
# ==========================================


def plot_prediction_full_and_zoom(model_name, df):
    """全期間とZoomの予測プロットを作成"""
    df["Date"] = pd.to_datetime(df["Date"])

    # ランダムに3銘柄選ぶ（毎回同じになるようにシード固定）
    np.random.seed(42)
    codes = df["code"].unique()
    if len(codes) == 0:
        return
    samples = np.random.choice(codes, min(3, len(codes)), replace=False)

    # --- A. 全期間 (Full) ---
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 3 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, code in zip(axes, samples):
        data = df[df["code"] == code].sort_values("Date")
        name = data["Name"].iloc[0] if "Name" in data.columns else str(code)

        ax.plot(data["Date"], data["Actual"], label="Actual", color="black", alpha=0.5, linewidth=1)
        ax.plot(data["Date"], data["Pred"], label="Pred", color="blue", linestyle="--", alpha=0.8, linewidth=1)
        ax.set_title(f"[{model_name}] {code} {name} (Full)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y/%m"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_pred_full.png", dpi=300)
    plt.close()

    # --- B. 直近拡大 (Zoom) ---
    fig, axes = plt.subplots(len(samples), 1, figsize=(10, 3 * len(samples)), sharex=False)
    if len(samples) == 1:
        axes = [axes]

    for ax, code in zip(axes, samples):
        data = df[df["code"] == code].sort_values("Date").iloc[-100:]  # Last 100 days
        if len(data) == 0:
            continue
        name = data["Name"].iloc[0] if "Name" in data.columns else str(code)

        ax.plot(data["Date"], data["Actual"], label="Actual", color="black", marker=".", markersize=3, alpha=0.5)
        ax.plot(
            data["Date"], data["Pred"], label="Pred", color="red", linestyle="--", marker=".", markersize=3, alpha=0.8
        )
        ax.set_title(f"[{model_name}] {code} {name} (Zoom 100days)", fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_pred_zoom.png", dpi=300)
    plt.close()


def plot_scatter(model_name, df):
    """散布図を作成"""
    plt.figure(figsize=(5, 5))

    # リターン計算
    actual_ret = (df["Actual"] - df["Current"]) / df["Current"]
    pred_ret = df["Pred_Return"]

    # 範囲制限（外れ値対策）
    limit = 0.2  # +/- 20%

    plt.scatter(actual_ret, pred_ret, alpha=0.2, s=5, color="navy")
    plt.plot([-1, 1], [-1, 1], "r--", alpha=0.5)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)

    plt.title(f"{model_name} Scatter", fontsize=12)
    plt.xlabel("Actual Return")
    plt.ylabel("Predicted Return")
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_scatter.png", dpi=300)
    plt.close()


def plot_feature_importance(model_name):
    """特徴量重要度があればプロット"""
    csv_path = RESULT_DIR / f"{model_name}_feature_importance.csv"
    if not csv_path.exists():
        return  # ファイルがなければスキップ

    df = pd.read_csv(csv_path).sort_values("Importance", ascending=False).head(20)

    plt.figure(figsize=(8, 6))
    sns.barplot(x="Importance", y="Feature", data=df, palette="viridis")
    plt.title(f"{model_name} Feature Importance", fontsize=12)
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{model_name}_feature_importance.png", dpi=300)
    plt.close()


# ==========================================
# 3. メイン処理
# ==========================================
if __name__ == "__main__":
    print("--- 全モデルの補足図表生成を開始 ---")

    for model in ALL_MODELS:
        print(f"Processing: {model}...")

        # 1. 予測データの読み込み
        pred_file = RESULT_DIR / f"predictions_{model}.csv"
        if pred_file.exists():
            df_pred = pd.read_csv(pred_file)
            # 予測プロット
            plot_prediction_full_and_zoom(model, df_pred)
            # 散布図
            plot_scatter(model, df_pred)
        else:
            print(f"  -> Skip predictions (not found): {pred_file}")

        # 2. 特徴量重要度（あれば）
        plot_feature_importance(model)

    print("\n完了しました。")
