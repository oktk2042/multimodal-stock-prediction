import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(font="MS Gothic")

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(".").resolve()
TARGET_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
OUTPUT_DIR = TARGET_DIR

# 元データ
INPUT_FILE = PROJECT_ROOT / "1_data" / "processed" / "dataset_for_modeling_top200_final.csv"

TRANSACTION_COST = 0.001

# 新しいファイル名に対応するマッピング
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


def load_all_predictions():
    print(f"Loading predictions from: {TARGET_DIR}")
    pred_files = list(TARGET_DIR.glob("predictions_*.csv"))

    model_preds = {}
    print(f"Found {len(pred_files)} prediction files.")

    for f in pred_files:
        # ファイル名からモデルキーを抽出 (predictions_RidgeRegression.csv -> RidgeRegression)
        model_key = f.stem.replace("predictions_", "")

        try:
            df = pd.read_csv(f)
            df["Date"] = pd.to_datetime(df["Date"])
            model_preds[model_key] = df
            print(f"Loaded: {model_key} ({len(df)} rows)")
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    return model_preds


def calculate_max_drawdown(cumulative_returns):
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    return drawdown.min()


def run_backtest(df, model_name):
    # ポジション計算: 翌日の予測リターンがプラスなら買い(1), マイナスなら売り(-1)
    # ※予測は "Pred_Return" 列にあると想定

    # 日付でソート
    df = df.sort_values(["code", "Date"])

    # 銘柄ごとのリターン計算
    results = []

    for code, group in df.groupby("code"):
        group = group.copy()

        # 予測に基づくシグナル
        group["Signal"] = np.where(group["Pred_Return"] > 0, 1, -1)

        # 実現リターン (Actual Return = (Actual - Current)/Current)
        # ※ファイルによってはカラム名が異なる場合があるため確認
        if "Actual" in group.columns and "Current" in group.columns:
            group["Actual_Return"] = (group["Actual"] - group["Current"]) / group["Current"]
        else:
            # カラムがない場合はスキップ
            continue

        # 戦略リターン (コスト考慮)
        # Signalが前日と変わった場合にコスト発生
        group["Cost"] = group["Signal"].diff().abs().fillna(1) * TRANSACTION_COST
        group["Strategy_Return"] = group["Signal"] * group["Actual_Return"] - group["Cost"]

        results.append(group[["Date", "Strategy_Return"]])

    if not results:
        return None

    # 全銘柄の平均リターン（ポートフォリオ）
    all_res = pd.concat(results)
    portfolio = all_res.groupby("Date")["Strategy_Return"].mean()

    # 累積リターン
    cumulative_returns = (1 + portfolio).cumprod()

    # 指標算出
    total_return = cumulative_returns.iloc[-1] - 1
    daily_std = portfolio.std()
    sharpe_ratio = (portfolio.mean() / daily_std) * np.sqrt(252) if daily_std != 0 else 0
    max_dd = calculate_max_drawdown(cumulative_returns)

    return {
        "Model": NAME_MAP.get(model_name, model_name),
        "Total Return": total_return,
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_dd,
    }


def main():
    print("--- Starting Financial Analysis ---")

    model_preds = load_all_predictions()
    if not model_preds:
        print("No prediction data found.")
        return

    backtest_results = []
    summary_data = []

    for model_key, df in tqdm(model_preds.items(), desc="Analyzing Models"):
        # バックテスト
        bt_res = run_backtest(df, model_key)
        if bt_res:
            backtest_results.append(bt_res)

            # 予測精度指標 (R2, Accuracy)
            if "Actual" in df.columns and "Current" in df.columns:
                actual_ret = (df["Actual"] - df["Current"]) / df["Current"]
                pred_ret = df["Pred_Return"]

                # 方向正解率
                acc = accuracy_score(np.sign(actual_ret), np.sign(pred_ret))
                # R2 Score (Return)
                r2 = r2_score(actual_ret, pred_ret)

                formal_name = NAME_MAP.get(model_key, model_key)
                summary_data.append(
                    {
                        "Model": formal_name,
                        "Accuracy": acc,
                        "R2_Return": r2,
                        "Sharpe Ratio": bt_res["Sharpe Ratio"],
                        "Total Return": bt_res["Total Return"],
                        "Max Drawdown": bt_res["Max Drawdown"],
                    }
                )

    # 結果の保存
    if summary_data:
        df_summary = pd.DataFrame(summary_data).sort_values("Sharpe Ratio", ascending=False)
        save_path = OUTPUT_DIR / "final_financial_summary.csv"
        df_summary.to_csv(save_path, index=False)

        print("\n[Analysis Result]")
        print(df_summary)
        print(f"\nSaved summary to: {save_path}")

        # Sharpe Ratioの比較グラフ
        plt.figure(figsize=(10, 6))
        sns.barplot(x="Sharpe Ratio", y="Model", data=df_summary, palette="viridis")
        plt.title("Risk-Adjusted Return (Sharpe Ratio) Comparison")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "final_sharpe_comparison.png")
        print(f"Saved plot to: {OUTPUT_DIR / 'final_sharpe_comparison.png'}")


if __name__ == "__main__":
    main()
