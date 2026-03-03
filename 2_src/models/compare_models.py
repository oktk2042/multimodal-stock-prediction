import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

# ==========================================
# 設定: 論文掲載用 シンプルスタイル
# ==========================================
# 背景白、グリッドあり、文字大きめ
sns.set_theme(style="whitegrid", context="paper", font_scale=1.4)
plt.rcParams["font.family"] = "MS Gothic"  # 日本語フォント
plt.rcParams["figure.figsize"] = (12, 8)  # 標準的な比率
warnings.filterwarnings("ignore")


class Config:
    # 【修正】実行している場所(カレントディレクトリ)をルートにする
    PROJECT_ROOT = Path(".").resolve()

    # 探索するフォルダ (カレントディレクトリを優先的に探す設定に変更)
    REPORT_DIRS = [
        PROJECT_ROOT,  # カレントディレクトリ
        PROJECT_ROOT / "3_reports" / "phase3_production_deep_strict",
        PROJECT_ROOT / "3_reports" / "phase3_production_ridge_strict",
        PROJECT_ROOT / "3_reports" / "phase3_production_lgbm_strict",
    ]

    # 出力先
    OUTPUT_DIR = PROJECT_ROOT

    # 正式名称への変換マップ
    NAME_MAP = {
        "Ridge": "Ridge Regression",
        "LightGBM": "LightGBM",
        "LSTM": "Attention-LSTM",
        "Transformer": "Vanilla Transformer",
        "DLinear": "DLinear",
        "PatchTST": "PatchTST",
        "iTransformer": "iTransformer",
        "FusionTransformer": "Multi-Modal Gated Transformer (Ours)",
    }

    # 論文用の表示順序 (ML -> DL -> SOTA -> Proposed)
    MODEL_ORDER = [
        "Ridge Regression",
        "LightGBM",
        "Vanilla Transformer",
        "Attention-LSTM",
        "DLinear",
        "PatchTST",
        "iTransformer",
        "Multi-Modal Gated Transformer (Ours)",
    ]


def calculate_metrics_df(df_res):
    """
    全7指標を計算する関数
    """
    required = ["Actual", "Pred", "Current"]
    if not all(c in df_res.columns for c in required):
        return None

    # 欠損除去
    df_res = df_res.dropna(subset=required)
    if len(df_res) == 0:
        return None

    # データ抽出
    y_true_price = df_res["Actual"].values
    y_pred_price = df_res["Pred"].values
    y_curr_price = df_res["Current"].values

    # リターン予測値
    if "Pred_Return" in df_res.columns:
        pred_return = df_res["Pred_Return"].values
    else:
        pred_return = np.log(y_pred_price / y_curr_price)

    # 実測リターン
    with np.errstate(divide="ignore", invalid="ignore"):
        actual_return = np.log(y_true_price / y_curr_price)

    # 無効値除去
    mask = np.isfinite(actual_return) & np.isfinite(pred_return)
    if not np.any(mask):
        return None

    y_true_price = y_true_price[mask]
    y_pred_price = y_pred_price[mask]
    y_curr_price = y_curr_price[mask]
    pred_return = pred_return[mask]
    actual_return = actual_return[mask]

    # 1. 誤差指標
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae = mean_absolute_error(y_true_price, y_pred_price)

    # 2. Accuracy
    diff_true = y_true_price - y_curr_price
    diff_pred = y_pred_price - y_curr_price
    with np.errstate(divide="ignore", invalid="ignore"):
        accuracy = accuracy_score(np.sign(diff_true), np.sign(diff_pred)) * 100

    # 3. R2
    r2_price = r2_score(y_true_price, y_pred_price)
    r2_return = r2_score(actual_return, pred_return)

    # 4. MAPE
    mask_mape = y_true_price != 0
    if np.sum(mask_mape) > 0:
        mape = np.mean(np.abs((y_true_price[mask_mape] - y_pred_price[mask_mape]) / y_true_price[mask_mape])) * 100
    else:
        mape = np.nan

    # 5. Corr
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


def plot_bar_chart(df, x_col, y_col, title, filename, output_dir):
    """論文掲載用 シンプルな棒グラフ"""
    plt.figure(figsize=(12, 8))

    # シンプルで見やすい配色のパレット (viridis) を使用
    # hueを指定して警告を回避しつつ、legend=Falseで凡例を非表示に
    ax = sns.barplot(x=x_col, y=y_col, data=df, palette="viridis", hue=x_col, legend=False)

    plt.title(title, fontsize=18, pad=15, fontweight="bold")
    plt.ylabel(y_col, fontsize=16)
    plt.xlabel("", fontsize=14)

    # X軸ラベルの回転
    plt.xticks(rotation=30, ha="right", fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # 範囲調整
    y_min, y_max = df[y_col].min(), df[y_col].max()
    margin = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 1.0
    if margin == 0:
        margin = 1.0
    plt.ylim(y_min - margin, y_max + margin)

    # 数値ラベル (シンプルに黒文字)
    for p in ax.patches:
        height = p.get_height()
        if np.isfinite(height):
            offset = 10 if height >= 0 else -20
            va = "bottom" if height >= 0 else "top"

            # 桁数の調整 (Accuracy/MAPEは小数点1桁, 他は4桁)
            if "Accuracy" in y_col or "MAPE" in y_col:
                label = f"{height:.1f}%"
            else:
                label = f"{height:.4f}"

            ax.annotate(
                label,
                (p.get_x() + p.get_width() / 2.0, height),
                ha="center",
                va=va,
                xytext=(0, offset),
                textcoords="offset points",
                fontsize=12,
                color="black",
            )

    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=300)
    plt.close()
    print(f"[Output] 画像を保存しました: {filename}")


def main():
    print("=" * 60)
    print(" 論文用モデル比較・集計ツール (修正版)")
    print("=" * 60)

    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    processed_keys = set()

    # フォルダ巡回
    for report_dir in Config.REPORT_DIRS:
        if not report_dir.exists():
            continue

        # ファイル検索
        pred_files = list(report_dir.glob("predictions_*.csv"))
        print(f"[Search] {report_dir}: {len(pred_files)} files found.")

        for p_file in pred_files:
            raw_key = p_file.stem.replace("predictions_", "")

            if raw_key in processed_keys:
                continue

            try:
                df = pd.read_csv(p_file)
                metrics = calculate_metrics_df(df)
                if metrics is not None:
                    formal_name = Config.NAME_MAP.get(raw_key, raw_key)
                    metrics["Model"] = formal_name

                    summary.append(metrics)
                    processed_keys.add(raw_key)
            except Exception as e:
                print(f"Error reading {p_file}: {e}")

    if not summary:
        print("[Error] 結果ファイルが見つかりませんでした。")
        print("※ predictions_*.csv ファイルが同じフォルダにあるか確認してください。")
        return

    summary_df = pd.DataFrame(summary)

    # 順序の適用
    order_list = Config.MODEL_ORDER + [m for m in summary_df["Model"].unique() if m not in Config.MODEL_ORDER]
    summary_df["Model"] = pd.Categorical(summary_df["Model"], categories=order_list, ordered=True)
    summary_df = summary_df.sort_values("Model")

    # 1. CSV保存 (全指標)
    csv_path = Config.OUTPUT_DIR / "final_model_comparison.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"\n[Output] 集計CSVを保存しました: {csv_path}")

    # 2. コンソール表示
    # 指定されたカラムのみ表示 (存在しないものはスキップ)
    disp_cols = ["Model", "R2_Return", "Accuracy", "R2_Price", "RMSE", "MAE", "Corr", "MAPE"]
    disp_cols = [c for c in disp_cols if c in summary_df.columns]

    print("\n【最終比較結果】")
    print(summary_df[disp_cols].to_string(index=False))

    # 3. 画像保存
    plot_bar_chart(
        summary_df, "Model", "Accuracy", "Model Accuracy Comparison", "final_accuracy_comparison.png", Config.OUTPUT_DIR
    )

    plot_bar_chart(
        summary_df,
        "Model",
        "R2_Return",
        "Model Return R2 Score Comparison",
        "final_r2_return_comparison.png",
        Config.OUTPUT_DIR,
    )

    print("\n[Done] 全処理が完了しました。")


if __name__ == "__main__":
    main()
