from pathlib import Path

import pandas as pd

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"


def extract_news_examples():
    # CSV読み込み
    df = pd.read_csv(DATA_DIR / "sentiment_noise_check_list.csv")

    # 1. ポジティブ例（スコアが高い順）
    print("\n=== ポジティブニュース例 (Top 3) ===")
    top_pos = df.sort_values("News_Sentiment", ascending=False).head(3)
    for _, row in top_pos.iterrows():
        print(f"Date: {row['Date']} | Score: {row['News_Sentiment']:.4f}")
        print(f"Title: {row['Title']}")
        print("-" * 30)

    # 2. ネガティブ例（スコアが低い順）
    print("\n=== ネガティブニュース例 (Bottom 3) ===")
    top_neg = df.sort_values("News_Sentiment", ascending=True).head(3)
    for _, row in top_neg.iterrows():
        print(f"Date: {row['Date']} | Score: {row['News_Sentiment']:.4f}")
        print(f"Title: {row['Title']}")
        print("-" * 30)


if __name__ == "__main__":
    extract_news_examples()
