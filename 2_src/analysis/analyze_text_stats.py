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

# 分析対象ファイルとカラム設定
FILES = {
    "News (RSS)": {
        "path": PROCESSED_DIR / "collected_news_historical_full.csv",
        "date_col": "Date",
        "code_col": "Code",
        "text_col": "Title",
    },
    "EDINET (Reports)": {
        "path": DATA_DIR / "edinet_reports" / "00_metadata" / "metadata_2018_2025_all.csv",
        "date_col": "submitDateTime",
        "code_col": "secCode",
        "text_col": "docDescription",
    },
}

# グラフ設定 (日本語フォント)
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def analyze_text_data_final():
    stats = []
    daily_counts = {}
    stock_counts = {}

    for source_name, config in FILES.items():
        file_path = config["path"]
        if not file_path.exists():
            print(f"⚠ Skip: {source_name} (File not found: {file_path})")
            continue

        print(f"Analyzing {source_name}...")

        try:
            # DtypeWarning回避
            df = pd.read_csv(file_path, low_memory=False)

            # カラム名の確認
            date_col = config["date_col"]
            code_col = config["code_col"]

            if date_col not in df.columns:
                print(f"  Error: Column '{date_col}' not found. Columns: {list(df.columns)}")
                continue

            # --- データ加工 ---
            # 1. 日付変換
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col])

            # 期間フィルタ (2020-2025)
            df = df[(df[date_col] >= "2020-01-01") & (df[date_col] <= "2025-12-31")]

            if df.empty:
                print(f"  Warning: No data found in 2020-2025 range for {source_name}")
                continue

            # 2. 銘柄コード処理
            if code_col in df.columns:
                # 文字列化
                df[code_col] = df[code_col].astype(str)
                # EDINETの場合、5桁(末尾0)を4桁に短縮 (例: 72030 -> 7203)
                if source_name == "EDINET (Reports)":
                    df["clean_code"] = df[code_col].str[:4]
                else:
                    # Newsの場合、数字4桁を抽出
                    df["clean_code"] = df[code_col].str.extract(r"(\d{4})")[0]

            # 3. 統計量計算
            count = len(df)
            avg_len = 0
            text_col = config["text_col"]
            if text_col in df.columns:
                avg_len = df[text_col].astype(str).str.len().mean()

            stats.append(
                {
                    "Source": source_name,
                    "Total Count": count,
                    "Start Date": df[date_col].min().date(),
                    "End Date": df[date_col].max().date(),
                    "Avg Length (Chars)": int(avg_len),
                }
            )

            # 4. 月次集計用
            df["Month"] = df[date_col].dt.to_period("M")
            daily_counts[source_name] = df["Month"].value_counts().sort_index()

            # 5. 銘柄別集計用 (Newsのみ)
            if source_name == "News (RSS)" and "clean_code" in df.columns:
                stock_counts[source_name] = df["clean_code"].value_counts()

        except Exception as e:
            print(f"  Error analyzing {source_name}: {e}")

    # --- 結果出力 ---
    if not stats:
        print("\nNo statistics generated.")
        return

    # 1. 統計テーブル
    df_stats = pd.DataFrame(stats)
    print("\n=== Data Statistics ===")
    print(df_stats)
    df_stats.to_csv(OUTPUT_DIR / "text_data_statistics.csv", index=False)

    # 2. 時系列グラフ (月次)
    if daily_counts:
        plt.figure(figsize=(12, 6))
        for source, series in daily_counts.items():
            series.index = series.index.to_timestamp()
            # 軸を分ける（件数の桁が違うため）
            if source == "EDINET (Reports)":
                plt.plot(series.index, series.values, label=f"{source} (Left)", marker="x", linestyle="--")
            else:
                plt.plot(series.index, series.values, label=f"{source} (Left)", marker="o")

        plt.title("テキストデータ発生件数の推移 (2020-2025)", fontsize=16)
        plt.ylabel("件数 / 月", fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "text_data_monthly_trend.png", dpi=300)
        print(f"Saved: {OUTPUT_DIR / 'text_data_monthly_trend.png'}")

    # 3. 銘柄別ヒストグラム (Newsのみ)
    if "News (RSS)" in stock_counts:
        plt.figure(figsize=(10, 6))
        sns.histplot(stock_counts["News (RSS)"], bins=50, kde=False, color="skyblue")
        plt.title("銘柄ごとのニュース記事数分布 (情報のスパース性)", fontsize=16)
        plt.xlabel("期間中総記事数", fontsize=14)
        plt.ylabel("銘柄数", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / "news_sparsity_distribution.png", dpi=300)
        print(f"Saved: {OUTPUT_DIR / 'news_sparsity_distribution.png'}")


if __name__ == "__main__":
    analyze_text_data_final()
