from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"
plt.rcParams["axes.unicode_minus"] = False

# 設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data for Volatility Analysis...")
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE, dtype={"Code": str, "code": str})
    # カラム名の統一
    if "Code" in df.columns:
        df.rename(columns={"Code": "code"}, inplace=True)

    # ターゲットカラムの特定
    target_score_col = "Fin_FinBERT_Score" if "Fin_FinBERT_Score" in df.columns else "FinBERT_Score"
    print(f"Using Score Column: {target_score_col}")

    # ---------------------------------------------------------
    # 1. ニュース判定ロジックの修正
    # ---------------------------------------------------------
    # データセットはForward Fillされているため、「0以外」ではなく
    # 「前日から値が変化した日」をイベント発生日とする

    # 銘柄ごとにソート
    df.sort_values(["code", "Date"], inplace=True)

    # 前日のスコアを取得 (銘柄ごとにシフト)
    df["Prev_Score"] = df.groupby("code")[target_score_col].shift(1).fillna(0)

    # 判定: (スコアが0ではない) AND (前日と値が違う)
    df["Has_News"] = (df[target_score_col] != 0) & (df[target_score_col] != df["Prev_Score"])

    # ニュースの頻度確認
    news_count = df["Has_News"].sum()
    total_count = len(df)
    print(f"\nTotal Rows: {total_count}")
    print(f"News Events (True): {news_count} ({news_count / total_count:.2%})")
    print(f"No News (False): {total_count - news_count}")

    if news_count == 0:
        print("Warning: No news events detected. Check the score column values.")
        return

    # ---------------------------------------------------------
    # 2. ボラティリティ分析
    # ---------------------------------------------------------
    # 日次リターンの絶対値（ボラティリティの代理変数）
    # Closeに0や欠損が含まれる場合の対策
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])

    df["Log_Return_Abs"] = (
        df.groupby("code")["Close"].apply(lambda x: np.abs(np.log(x / x.shift(1)))).reset_index(level=0, drop=True)
    )

    print("\n--- Impact on Volatility (Absolute Daily Return) ---")
    stats_vol = df.groupby("Has_News")["Log_Return_Abs"].mean()
    print(stats_vol)

    # グラフ作成
    plt.figure(figsize=(8, 6))
    # 【修正】hueを指定して警告を解消
    sns.barplot(x=stats_vol.index, y=stats_vol.values, hue=stats_vol.index, palette="viridis", legend=False)
    plt.title("ニュース有無による日次変動幅（絶対値）の違い")
    plt.xlabel("ニュース有無 (False=なし, True=あり)")
    plt.ylabel("対数収益率の絶対値 (平均)")
    plt.grid(axis="y", alpha=0.3)

    # 数値を棒グラフの上に表示
    for i, v in enumerate(stats_vol.values):
        plt.text(i, v, f"{v:.4f}", ha="center", va="bottom")

    plt.savefig(OUTPUT_DIR / "impact_volatility.png")
    plt.close()

    # ---------------------------------------------------------
    # 3. 出来高分析
    # ---------------------------------------------------------
    if "Volume" in df.columns:
        # 0を除外して対数化
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce").fillna(0)
        df["Log_Volume"] = np.log(df["Volume"].replace(0, np.nan))

        print("\n--- Impact on Volume ---")
        stats_vol_mean = df.groupby("Has_News")["Log_Volume"].mean()
        print(stats_vol_mean)

        plt.figure(figsize=(8, 6))
        # 【修正】hueを指定して警告を解消
        sns.boxplot(
            data=df, x="Has_News", y="Log_Volume", hue="Has_News", palette="Pastel1", showfliers=False, legend=False
        )
        plt.title("ニュース有無による出来高(Log)の分布")
        plt.xlabel("ニュース有無")
        plt.ylabel("対数出来高")
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(OUTPUT_DIR / "impact_volume_boxplot.png")
        plt.close()

    print(f"\nAnalysis saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
