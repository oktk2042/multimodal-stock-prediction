from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_DIR = DATA_DIR / "final_datasets_yearly"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "data_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定（必要に応じて変更してください）
# sns.set(font='IPAGothic')


def analyze_distributions():
    print("--- 統合データの分布分析 ---")

    # 1. データ読み込み
    all_files = sorted(list(INPUT_DIR.glob("final_data_*.csv")))
    if not all_files:
        print("エラー: ファイルが見つかりません。")
        return

    print(f"対象ファイル数: {len(all_files)}")
    df_list = []
    for f in tqdm(all_files, desc="Loading"):
        try:
            # メモリ節約のため必要な列だけ... と言いたいが、異常チェックのため全列読む
            df = pd.read_csv(f, low_memory=False)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not df_list:
        return

    df_all = pd.concat(df_list, ignore_index=True)
    print(f"総行数: {len(df_all):,}")

    # 2. 異常値チェック (株価)
    print("\n=== 株価 (Close) の分布チェック ===")
    print(df_all["Close"].describe())

    # 閾値を超える株価の確認 (例: 10,000,000円)
    high_price_threshold = 10000000
    high_prices = df_all[df_all["Close"] > high_price_threshold]
    if not high_prices.empty:
        print(f"\n⚠️ 株価が {high_price_threshold:,} 円を超えるデータが {len(high_prices)} 件あります。")
        print(high_prices[["Date", "code", "Name", "Close"]].head(10).to_string(index=False))
    else:
        print(f"\n✅ 株価はすべて {high_price_threshold:,} 円以下です。")

    # 3. 異常値チェック (売上高)
    print("\n=== 売上高 (NetSales) の分布チェック ===")
    zero_sales = df_all[df_all["NetSales"] == 0]
    print(f"売上高が0の行数: {len(zero_sales):,} ({len(zero_sales) / len(df_all) * 100:.1f}%)")

    # 4. 感情スコアの分布チェック (-1 ~ 1)
    print("\n=== 感情スコア (Sentiment) の分布チェック ===")
    sentiment_cols = ["FinBERT_Score", "News_Sentiment", "Market_Sentiment"]

    for col in sentiment_cols:
        if col in df_all.columns:
            print(f"\n[{col}]")
            print(df_all[col].describe())

            # 範囲外チェック
            out_of_range = df_all[(df_all[col] < -1) | (df_all[col] > 1)]
            if not out_of_range.empty:
                print(f"⚠️ {col} に -1 ~ 1 の範囲外の値が {len(out_of_range)} 件あります！")
                print(out_of_range[col].head())
            else:
                print(f"✅ {col} は正常範囲内 (-1 ~ 1) です。")

            # ヒストグラム保存
            plt.figure(figsize=(10, 6))
            sns.histplot(df_all[col], bins=50, kde=True)
            plt.title(f"Distribution of {col}")
            plt.xlabel("Score")
            plt.xlim(-1.1, 1.1)  # 範囲を固定
            plt.grid(True, alpha=0.3)
            save_path = OUTPUT_DIR / f"dist_{col}.png"
            plt.savefig(save_path)
            print(f"ヒストグラム保存: {save_path}")
            plt.close()


if __name__ == "__main__":
    analyze_distributions()
