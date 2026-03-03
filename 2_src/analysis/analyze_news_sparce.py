import warnings
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定: ファイルパス
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 1. ニュース統計データ (全銘柄の記事数)
NEWS_STATS_FILE = PROJECT_ROOT / "1_data" / "processed" / "news_stats_by_keyword.csv"

# 2. 予測結果ファイル (ここに含まれる銘柄だけを対象にする)
# ※ファイル名が異なる場合は修正してください
PRED_FILE = PROJECT_ROOT / "3_reports" / "final_consolidated_v2" / "predictions_MultiModalGatedTransformer.csv"

# 3. ニュース詳細データ (グラフ描画用)
NEWS_DETAIL_FILE = PROJECT_ROOT / "1_data" / "processed" / "collected_news_historical_full.csv"


def plot_valid_news_comparison():
    warnings.simplefilter("ignore")
    print("=== バックテスト対象銘柄からの自動選定を開始 ===")

    # ---------------------------------------------------------
    # 1. バックテスト対象銘柄 (Universe) の特定
    # ---------------------------------------------------------
    if not PRED_FILE.exists():
        print(f"エラー: 予測ファイルが見つかりません。\nパスを確認してください: {PRED_FILE}")
        return

    print("予測結果ファイルを読み込み中...")
    df_pred = pd.read_csv(PRED_FILE)
    # コードを文字列4桁に統一
    df_pred["str_code"] = df_pred["code"].astype(str).str.extract(r"(\d{4})")[0]
    valid_universe = set(df_pred["str_code"].unique())
    print(f"-> バックテスト対象銘柄数: {len(valid_universe)}")

    # ---------------------------------------------------------
    # 2. ニュース記事数データとの照合とフィルタリング
    # ---------------------------------------------------------
    if not NEWS_STATS_FILE.exists():
        print(f"エラー: ニュース統計ファイルが見つかりません: {NEWS_STATS_FILE}")
        return

    df_stats = pd.read_csv(NEWS_STATS_FILE)
    df_stats["str_code"] = df_stats["Code"].astype(str).str.extract(r"(\d{4})")[0]

    # 9999を除外
    df_stats = df_stats[df_stats["str_code"] != "9999"]
    df_valid = df_stats[df_stats["str_code"].isin(valid_universe)].copy()

    if df_valid.empty:
        print("エラー: バックテスト対象銘柄とニュースデータの銘柄が一致しませんでした。")
        return

    print(f"-> ニュースデータが存在する対象銘柄数: {len(df_valid)}")

    # ---------------------------------------------------------
    # 3. 6銘柄の自動選定 (High/Medium/Low)
    # ---------------------------------------------------------
    # 記事数でソート
    df_sorted = df_valid.sort_values("Articles", ascending=False).reset_index(drop=True)

    # 選定ロジック
    # High: 上位2つ
    high_stocks = df_sorted.head(2)

    # Medium: 中央値付近の2つ
    median_idx = len(df_sorted) // 2
    medium_stocks = df_sorted.iloc[median_idx - 1 : median_idx + 1]

    # Low: 下位2つ
    low_stocks = df_sorted.tail(2)

    # 選定結果を辞書化
    target_stocks = {
        "High Frequency": dict(
            zip(high_stocks["str_code"], high_stocks["Keyword"] + " (" + high_stocks["str_code"] + ")")
        ),
        "Medium Frequency": dict(
            zip(medium_stocks["str_code"], medium_stocks["Keyword"] + " (" + medium_stocks["str_code"] + ")")
        ),
        "Low Frequency": dict(zip(low_stocks["str_code"], low_stocks["Keyword"] + " (" + low_stocks["str_code"] + ")")),
    }

    print("\n【自動選定された比較対象銘柄】")
    for cat, stocks in target_stocks.items():
        print(f"--- {cat} ---")
        for code, name in stocks.items():
            count = df_valid[df_valid["str_code"] == code]["Articles"].values[0]
            print(f"  {name}: {count} articles")

    # ---------------------------------------------------------
    # 4. グラフ描画 (前回と同じロジック)
    # ---------------------------------------------------------
    if not NEWS_DETAIL_FILE.exists():
        print(f"エラー: 詳細ニュースデータが見つかりません: {NEWS_DETAIL_FILE}")
        return

    print("\nグラフ描画用データを読み込み中...")
    df_detail = pd.read_csv(NEWS_DETAIL_FILE, low_memory=False)
    date_col = "Date" if "Date" in df_detail.columns else "published"
    df_detail[date_col] = pd.to_datetime(df_detail[date_col], errors="coerce")
    df_detail = df_detail.dropna(subset=[date_col])
    df_detail = df_detail[(df_detail[date_col] >= "2020-01-01") & (df_detail[date_col] <= "2025-12-31")]
    if "Code" in df_detail.columns:
        df_detail["str_code"] = df_detail["Code"].astype(str).str.extract(r"(\d{4})")[0]

    # プロット
    sns.set(style="whitegrid")
    plt.rcParams["font.family"] = "MS Gothic"
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    all_months = pd.date_range(start="2020-01-01", end="2025-12-31", freq="M")  # pandas < 2.2なら 'M'
    colors = {"High Frequency": "royalblue", "Medium Frequency": "mediumseagreen", "Low Frequency": "salmon"}

    categories = ["High Frequency", "Medium Frequency", "Low Frequency"]

    for row_idx, category in enumerate(categories):
        stocks = target_stocks[category]
        color = colors[category]

        for col_idx, (code, name) in enumerate(stocks.items()):
            ax = axes[row_idx, col_idx]
            stock_data = df_detail[df_detail["str_code"] == code]

            monthly_counts = stock_data.resample("M", on=date_col).size()
            monthly_counts = monthly_counts.reindex(all_months, fill_value=0)

            ax.bar(monthly_counts.index, monthly_counts.values, width=20, color=color, alpha=0.8)
            ax.set_title(f"[{category}]\n{name}\nTotal: {int(monthly_counts.sum())}", fontsize=11, fontweight="bold")
            ax.set_ylabel("Monthly Articles")
            ax.grid(True, alpha=0.3)

            # Lowのスケール調整
            if category == "Low Frequency" and monthly_counts.max() < 10:
                ax.set_yticks(range(0, int(monthly_counts.max()) + 3, 2))

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
            ax.xaxis.set_major_locator(mdates.YearLocator())

    fig.suptitle("バックテスト対象銘柄におけるニュース頻度比較 (High vs Medium vs Low)", fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])

    save_path = "valid_news_frequency_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nグラフを保存しました: {save_path}")


if __name__ == "__main__":
    plot_valid_news_comparison()
