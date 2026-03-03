import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力ファイル
STATS_FILE = PROCESSED_DIR / "news_stats_by_keyword.csv"  # 今回のファイル
FULL_LOG_FILE = PROCESSED_DIR / "collected_news_historical_full.csv"  # ソース分析用

# グラフ設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def analyze_news_final():
    print("=== News Data Analysis Final ===")

    # ---------------------------------------------------------
    # 1. 銘柄別ランキング (news_stats_by_keyword.csv 使用)
    # ---------------------------------------------------------
    if STATS_FILE.exists():
        print(f"Loading stats from: {STATS_FILE}")
        df_stats = pd.read_csv(STATS_FILE)

        # 9999 (マクロ指標) を除外
        df_clean = df_stats[df_stats["Code"].astype(str) != "9999"].copy()

        # 記事数でソートしてTop 20
        top15 = df_clean.sort_values("Articles", ascending=False).head(20)

        # 表示用ラベル: "キーワード (コード)"
        top15["Label"] = top15.apply(lambda x: f"{x['Keyword']} ({x['Code']})", axis=1)

        print("\nTop 20 Stocks:")
        print(top15[["Code", "Keyword", "Articles"]])

        # グラフ描画
        plt.figure(figsize=(10, 8))
        sns.barplot(x=top15["Articles"], y=top15["Label"], hue=top15["Label"], legend=False, palette="magma")
        plt.title("銘柄別ニュース記事数ランキング (Top 20)", fontsize=14)
        plt.xlabel("記事数 (2020-2025)", fontsize=12)
        plt.ylabel("銘柄 (キーワード)", fontsize=12)
        plt.tight_layout()

        save_path = OUTPUT_DIR / "news_stock_ranking_refined.png"
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")
    else:
        print(f"File not found: {STATS_FILE}")

    # ---------------------------------------------------------
    # 2. ニュースソース分析 (collected_news_historical_full.csv 使用)
    # ---------------------------------------------------------
    if FULL_LOG_FILE.exists():
        print(f"\nLoading full logs from: {FULL_LOG_FILE}")
        # DtypeWarning回避
        df_full = pd.read_csv(FULL_LOG_FILE, low_memory=False)

        # タイトルからメディア名を抽出する関数
        def extract_publisher(title):
            if not isinstance(title, str):
                return "Unknown"
            parts = title.rsplit(" - ", 1)  # 末尾の " - メディア名" を探す
            if len(parts) > 1:
                pub = parts[-1].strip()
                return re.sub(r"\.\.\.$", "", pub)  # 末尾の...を除去
            return "Unknown"

        if "Title" in df_full.columns:
            print("Extracting publishers...")
            df_full["Publisher"] = df_full["Title"].apply(extract_publisher)

            # 集計 (UnknownとGoogle Newsを除外)
            pub_counts = (
                df_full[~df_full["Publisher"].isin(["Unknown", "Google News"])]["Publisher"].value_counts().head(15)
            )

            print("\nTop 15 Publishers:")
            print(pub_counts)

            # グラフ描画
            plt.figure(figsize=(10, 8))
            sns.barplot(x=pub_counts.values, y=pub_counts.index, hue=pub_counts.index, legend=False, palette="viridis")
            plt.title("ニュース提供元メディアの内訳 (Top 15)", fontsize=14)
            plt.xlabel("記事数", fontsize=12)
            plt.tight_layout()

            save_path = OUTPUT_DIR / "news_source_distribution.png"
            plt.savefig(save_path, dpi=300)
            print(f"Saved: {save_path}")
        else:
            print("Error: 'Title' column not found in full log.")
    else:
        print(f"\nSkip Source Analysis (File not found: {FULL_LOG_FILE})")


if __name__ == "__main__":
    analyze_news_final()
