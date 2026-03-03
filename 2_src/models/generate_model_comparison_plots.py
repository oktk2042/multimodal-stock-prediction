from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")

PROJECT_ROOT = Path(".").resolve()
TARGET_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
OUTPUT_DIR = TARGET_DIR
# 正式名称マップ
NAME_MAP = {
    "RidgeRegression": "Ridge Regression",
    "LightGBM": "LightGBM",
    "AttentionLSTM": "Attention-LSTM",
    "VanillaTransformer": "Vanilla Transformer",
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "iTransformer": "iTransformer",
    "MultiModalGatedTransformer": "Multi-Modal Gated Transformer (Ours)",
}


def main():
    print("--- Generating Comparison Plots ---")

    # final_model_comparison.csv を読み込む
    summary_path = TARGET_DIR / "final_model_comparison.csv"

    if not summary_path.exists():
        # なければ model_comparison_summary.csv を探す
        summary_path = TARGET_DIR / "model_comparison_summary.csv"

    if not summary_path.exists():
        print("Summary CSV not found.")
        return

    print(f"Loading summary from: {summary_path.name}")
    df = pd.read_csv(summary_path)

    # モデル名を正式名称に統一 (念のため)
    if "Model" not in df.columns:
        print("Column 'Model' not found.")
        return

    # ---------------------------------------------------------
    # 修正箇所: hue='Model', legend=False を追加して警告を解消
    # ---------------------------------------------------------

    # 1. Accuracy Comparison
    if "Accuracy" in df.columns:
        plt.figure(figsize=(10, 6))
        df_acc = df.sort_values("Accuracy", ascending=False)

        # 修正: x, y に加えて hue=yの変数, legend=False を指定
        sns.barplot(x="Accuracy", y="Model", hue="Model", data=df_acc, palette="viridis", legend=False)

        plt.title("Directional Accuracy Comparison")
        plt.xlabel("Accuracy (%)" if df["Accuracy"].max() > 1 else "Accuracy")
        plt.xlim(40, 60)  # 範囲調整
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "final_accuracy_comparison.png")
        print("Saved: final_accuracy_comparison.png")

    # 2. R2 (Return) Comparison
    if "R2_Return" in df.columns:
        plt.figure(figsize=(10, 6))
        df_r2 = df.sort_values("R2_Return", ascending=False)
        sns.barplot(x="R2_Return", y="Model", hue="Model", data=df_r2, palette="magma", legend=False)

        plt.title("R2 Score (Return Prediction) Comparison")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "final_r2_return_comparison.png")
        print("Saved: final_r2_return_comparison.png")

    # 3. RMSE Comparison
    if "RMSE" in df.columns:
        plt.figure(figsize=(10, 6))
        # RMSEは小さい方が良いので昇順
        df_rmse = df.sort_values("RMSE", ascending=True)
        sns.barplot(x="RMSE", y="Model", hue="Model", data=df_rmse, palette="rocket", legend=False)

        plt.title("RMSE Comparison (Lower is Better)")
        plt.xlim(550, 565)  # 範囲調整
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "final_rmse_comparison.png")
        print("Saved: final_rmse_comparison.png")


if __name__ == "__main__":
    main()
