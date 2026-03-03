from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力ファイル
# 1. モデル学習用データ (Top 200, 株価・スコア入り)
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
# 2. ニュースタイトル詳細リスト (sentiment_noise_check_list.csv)
NEWS_FILE = DATA_DIR / "sentiment_noise_check_list.csv"
# 3. 銘柄リスト (社名用)
LIST_FILE = DATA_DIR / "stock_sector_info.csv"  # または top200_stock_list_final.csv


def find_candidates():
    print("Loading data...")

    if not PRICE_FILE.exists():
        print(f"Error: {PRICE_FILE} not found.")
        return

    # 1. 株価データの読み込み & 5日後リターン計算
    df_price = pd.read_csv(PRICE_FILE, dtype={"Code": str, "code": str})
    # カラム名統一
    if "Code" in df_price.columns:
        df_price.rename(columns={"Code": "code"}, inplace=True)

    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_price.sort_values(["code", "Date"], inplace=True)

    # 【重要】5日後リターンを計算 (実験設定と統一)
    df_price["Return_5D"] = df_price.groupby("code")["Close"].transform(lambda x: np.log(x.shift(-5) / x))

    # 2. ニュース詳細データの読み込み
    if not NEWS_FILE.exists():
        print(f"Error: {NEWS_FILE} not found.")
        return

    df_news = pd.read_csv(NEWS_FILE, dtype={"Code": str, "code": str})
    if "Code" in df_news.columns:
        df_news.rename(columns={"Code": "code"}, inplace=True)
    df_news["Date"] = pd.to_datetime(df_news["Date"])

    # 3. データ結合 (株価データにある銘柄・日付のみ残す)
    # これにより Top 200 以外のニュースは除外される
    df_merged = pd.merge(df_news, df_price[["Date", "code", "Return_5D", "Name"]], on=["Date", "code"], how="inner")

    print(f"Matched News Events in Top 200: {len(df_merged)}")

    if len(df_merged) == 0:
        print("No matching news found. Check date formats or codes.")
        return

    # 4. ケース抽出ロジック
    # Case A: Positive Impact (Good News -> Price UP)
    # 条件: センチメント > 0.5 かつ 5日後リターン > 5%
    positive_cases = df_merged[(df_merged["News_Sentiment"] > 0.5) & (df_merged["Return_5D"] > 0.05)].sort_values(
        "Return_5D", ascending=False
    )

    # Case B: Negative Impact (Bad News -> Price DOWN)
    # 条件: センチメント < -0.5 かつ 5日後リターン < -5%
    negative_cases = df_merged[(df_merged["News_Sentiment"] < -0.5) & (df_merged["Return_5D"] < -0.05)].sort_values(
        "Return_5D", ascending=True
    )  # 下落率が大きい順

    # Case C: Surprise/Divergence (Good News -> Price DOWN)
    # 織り込み済み、または地合いが悪かったケース
    divergence_cases = df_merged[(df_merged["News_Sentiment"] > 0.7) & (df_merged["Return_5D"] < -0.05)].sort_values(
        "News_Sentiment", ascending=False
    )

    # 5. 結果出力
    def print_and_save(df, label, filename):
        if df.empty:
            print(f"\n=== {label} (0 cases) ===")
            return

        print(f"\n=== {label} (Top 10) ===")
        # 表示用カラム
        cols = ["Date", "code", "Name", "News_Sentiment", "Return_5D", "Title"]
        print(
            df[cols]
            .head(10)
            .to_string(
                index=False,
                formatters={
                    "News_Sentiment": "{:.2f}".format,
                    "Return_5D": "{:.2%}".format,
                    "Title": lambda x: x[:40] + "..." if len(str(x)) > 40 else str(x),
                },
            )
        )

        # CSV保存
        save_path = OUTPUT_DIR / filename
        df.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved to: {save_path}")

    print_and_save(positive_cases, "Positive Cases (Sentiment > 0.5 & Return > +5%)", "case_study_positive.csv")
    print_and_save(negative_cases, "Negative Cases (Sentiment < -0.5 & Return < -5%)", "case_study_negative.csv")
    print_and_save(divergence_cases, "Divergence Cases (Sentiment > 0.7 & Return < -5%)", "case_study_divergence.csv")


if __name__ == "__main__":
    find_candidates()
