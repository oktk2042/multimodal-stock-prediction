from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"

# 設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading data...")
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    df["Date"] = pd.to_datetime(df["Date"])

    # カラム名の確認
    target_score_col = "Fin_FinBERT_Score"
    if target_score_col not in df.columns and "FinBERT_Score" in df.columns:
        target_score_col = "FinBERT_Score"

    print(f"Using Sentiment Column: {target_score_col}")

    # 5日後リターンの計算（念のため再計算）
    df.sort_values(["Code", "Date"], inplace=True)
    df["Return_5D"] = df.groupby("Code")["Close"].transform(lambda x: np.log(x.shift(-5) / x))

    # ---------------------------------------------------------
    # 1. 真のニューススパース性（情報の更新頻度）確認
    # ---------------------------------------------------------
    # 前日と値が違う日 ＝ 新しいレポートが出てスコアが更新された日
    # (注意: 0埋めされている期間も考慮し、0以外で値が変わった日を検知)

    def count_changes(series):
        # 0以外の値が入っており、かつ前日と異なる場合をカウント
        condition = (series != 0) & (series != series.shift(1).fillna(0))
        return condition.sum()

    total_rows = len(df)
    # 銘柄ごとに更新回数をカウント
    news_counts = df.groupby("Code")[target_score_col].apply(count_changes)
    total_events = news_counts.sum()

    print("\n=== True News Sparsity (Update Frequency) ===")
    print(f"Total Rows (Trading Days): {total_rows}")
    print(f"Total News Updates (Events): {total_events}")
    print(f"Event Sparsity Ratio: {total_events / total_rows:.2%}")
    print(f"Average Updates per Stock: {news_counts.mean():.1f}")

    # ヒストグラム: 銘柄ごとのレポート更新回数
    plt.figure(figsize=(10, 6))
    sns.histplot(news_counts, bins=30, kde=False, color="navy")
    plt.title("Distribution of FinBERT Score Updates per Stock (True Events)")
    plt.xlabel("Number of Updates (Reports)")
    plt.ylabel("Number of Stocks")
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(OUTPUT_DIR / "news_frequency_dist_true.png")
    plt.close()

    # ---------------------------------------------------------
    # 2. ニュース感情スコアとリターンの相関関係
    # ---------------------------------------------------------
    # データセット全体での相関（Forward Fillされた状態での相関）
    df_valid = df.dropna(subset=["Return_5D"]).copy()

    if len(df_valid) > 0:
        # 相関係数
        corr = df_valid[[target_score_col, "Return_5D"]].corr().iloc[0, 1]
        print(f"\nCorrelation (FinBERT vs 5D Return): {corr:.4f}")

        # 散布図 (サンプリング)
        sample_n = min(10000, len(df_valid))
        df_sample = df_valid.sample(sample_n, random_state=42)

        plt.figure(figsize=(8, 8))
        sns.scatterplot(data=df_sample, x=target_score_col, y="Return_5D", alpha=0.3, s=15)
        plt.axhline(0, color="black", ls="--", lw=0.8)
        plt.axvline(0, color="black", ls="--", lw=0.8)
        plt.title(f"FinBERT Score vs 5-Day Return (Corr: {corr:.3f})")
        plt.xlabel("FinBERT Sentiment Score")
        plt.ylabel("5-Day Log Return")
        plt.savefig(OUTPUT_DIR / "correlation_news_return.png")
        plt.close()

        # 箱ひげ図 (警告修正版)
        def categorize_sentiment(score):
            if score > 0.1:
                return "Positive"
            if score < -0.1:
                return "Negative"
            return "Neutral"

        df_valid["Sentiment_Cat"] = df_valid[target_score_col].apply(categorize_sentiment)

        plt.figure(figsize=(8, 6))
        # hueにxと同じ変数を指定し、legend=Falseにする
        sns.boxplot(
            data=df_valid,
            x="Sentiment_Cat",
            y="Return_5D",
            hue="Sentiment_Cat",
            order=["Negative", "Neutral", "Positive"],
            palette="coolwarm",
            legend=False,
        )
        plt.title("5-Day Return Distribution by Sentiment Category")
        plt.ylim(-0.2, 0.2)
        plt.axhline(0, color="black", ls="-", lw=0.5)
        plt.savefig(OUTPUT_DIR / "return_by_sentiment_boxplot.png")
        plt.close()
    else:
        print("No valid data found.")

    print(f"\nAnalysis saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
