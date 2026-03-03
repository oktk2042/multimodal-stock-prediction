from pathlib import Path

# ==========================================
# 1. 設定 & 日本語フォント
# ==========================================
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Windows標準フォント
matplotlib.rcParams["font.family"] = "MS Gothic"
matplotlib.rcParams["axes.unicode_minus"] = False
sns.set(font="MS Gothic", style="whitegrid")

# パス設定 (2_src/analysis/ から見た相対パス)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力ファイル
NEWS_STATS_FILE = DATA_DIR / "news_stats_by_code.csv"
NEWS_MATRIX_FILE = DATA_DIR / "news_stats_monthly_matrix.csv"
SENTIMENT_TREND_FILE = DATA_DIR / "sentiment_monthly_trend.csv"
DATASET_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"

# ==========================================
# 2. 分析関数群
# ==========================================


def analyze_news_distribution():
    """ニュース記事数の分布（スパース性の確認）"""
    print("--- [1/6] ニュース分布の分析 ---")
    if not NEWS_STATS_FILE.exists():
        return

    df = pd.read_csv(NEWS_STATS_FILE)

    plt.figure(figsize=(10, 6))
    sns.histplot(df["Articles_Per_Month"], bins=40, kde=True, color="skyblue", edgecolor="black")
    plt.title("銘柄ごとの月間平均ニュース記事数分布")
    plt.xlabel("月間平均記事数")
    plt.ylabel("銘柄数")

    # 統計量の表示
    mean_val = df["Articles_Per_Month"].mean()
    median_val = df["Articles_Per_Month"].median()
    plt.axvline(mean_val, color="red", linestyle="--", label=f"平均: {mean_val:.1f}")
    plt.axvline(median_val, color="green", linestyle="-", label=f"中央値: {median_val:.1f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_news_distribution_hist.png", dpi=300)
    plt.close()


def analyze_news_trend():
    """ニュース数の時系列推移"""
    print("--- [2/6] ニュース時系列推移の分析 ---")
    if not NEWS_MATRIX_FILE.exists():
        return

    df = pd.read_csv(NEWS_MATRIX_FILE)
    # 横持ち(Wide)を縦持ち(Long)に変換して集計
    date_cols = [c for c in df.columns if c != "Code"]

    # 全銘柄の合計を計算
    total_news = df[date_cols].sum()
    total_news.index = pd.to_datetime(total_news.index)

    plt.figure(figsize=(12, 6))
    plt.plot(total_news.index, total_news.values, marker=".", linestyle="-", color="navy", alpha=0.7)
    plt.title("全銘柄の月間ニュース記事数推移")
    plt.xlabel("年月")
    plt.ylabel("総記事数")
    plt.grid(True, alpha=0.3)

    # 重要なイベントに注釈を入れる例（必要に応じて調整）
    # plt.annotate('コロナショック', xy=(pd.Timestamp('2020-03-01'), total_news.loc['2020-03-01']), ...)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_news_monthly_trend.png", dpi=300)
    plt.close()


def analyze_sentiment_trend():
    """センチメントスコアの推移"""
    print("--- [3/6] センチメント推移の分析 ---")
    if not SENTIMENT_TREND_FILE.exists():
        return

    df = pd.read_csv(SENTIMENT_TREND_FILE)
    df["YearMonth"] = pd.to_datetime(df["YearMonth"])

    plt.figure(figsize=(12, 6))
    plt.plot(
        df["YearMonth"],
        df["Individual_Avg_Sentiment"],
        label="個別銘柄平均 (Individual Avg)",
        color="blue",
        linewidth=2,
    )

    # マクロセンチメントがあれば描画
    macro_col = [c for c in df.columns if "Macro" in c][0]
    plt.plot(
        df["YearMonth"],
        df[macro_col],
        label="マクロセンチメント (Market Overall)",
        color="orange",
        linestyle="--",
        alpha=0.8,
    )

    plt.axhline(0, color="black", linewidth=0.5)
    plt.title("市場センチメントの月次推移 (FinBERT Score)")
    plt.xlabel("年月")
    plt.ylabel("平均センチメントスコア (-1 to 1)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_sentiment_trend.png", dpi=300)
    plt.close()


def analyze_dataset_stats():
    """学習用データセットを用いた詳細分析"""
    if not DATASET_FILE.exists():
        print(f"スキップ: {DATASET_FILE} が見つかりません。")
        return

    print("--- データセット読み込み中（時間がかかる場合があります）... ---")
    # 必要な列だけ読み込むと高速化
    cols = [
        "Date",
        "code",
        "Target_Return_5D",
        "Log_Return",
        "Volatility_20D",
        "Volume",
        "News_Sentiment",
        "FinBERT_Score",
        "RSI_14",
        "MA_Gap_25D",
    ]
    # 全列読み込んでしまう
    df = pd.read_csv(DATASET_FILE, low_memory=False)

    print(f"データロード完了: {len(df):,} 行")

    # ----------------------------------------------------
    # 4. センチメントスコアの分布比較
    # ----------------------------------------------------
    print("--- [4/6] センチメント分布の分析 ---")
    plt.figure(figsize=(10, 6))

    # 0を除いたニュースセンチメント（発生日のみ）
    news_scores = df[df["News_Sentiment"] != 0]["News_Sentiment"]
    # 決算スコア
    fin_scores = df[df["FinBERT_Score"] != 0]["FinBERT_Score"]

    sns.kdeplot(news_scores, label="日次ニュース (News)", fill=True, color="blue", alpha=0.3)
    sns.kdeplot(fin_scores, label="決算短信 (Financials)", fill=True, color="green", alpha=0.3)

    plt.title("ニュースと決算情報の感情スコア分布比較")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Density")
    plt.xlim(-1, 1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_sentiment_distribution.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 5. 特徴量の相関ヒートマップ
    # ----------------------------------------------------
    print("--- [5/6] 相関ヒートマップの作成 ---")
    # 相関を見る特徴量
    corr_cols = [
        "Target_Return_5D",
        "Log_Return",
        "Volatility_20D",
        "News_Sentiment",
        "FinBERT_Score",
        "RSI_14",
        "MA_Gap_25D",
        "Volume",
    ]
    # 欠損除去して計算
    corr_mat = df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.5, vmax=0.5, center=0)
    plt.title("主要特徴量とリターンの相関行列")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_correlation_heatmap.png", dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 6. ニュース有無によるインパクト分析 (Boxplot) ★重要★
    # ----------------------------------------------------
    print("--- [6/6] ニュース有無によるインパクト分析 ---")

    # ニュースありフラグ
    df["Has_News"] = df["News_Sentiment"] != 0

    # ボラティリティへの影響
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="Has_News", y="Volatility_20D", data=df, showfliers=False, palette="Set2")
    plt.title("ニュース有無とボラティリティの関係")
    plt.xlabel("ニュースの有無 (False=なし, True=あり)")
    plt.ylabel("20日ボラティリティ")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "analysis_news_impact_volatility.png", dpi=300)
    plt.close()

    # 出来高への影響 (対数出来高を使用している前提)
    if "Volume" in df.columns:
        # 0やマイナスを除外して対数化（もし対数化されていなければ）
        # データセットが既に対数化されているかは分布で判断するか、カラム名で判断
        # ここでは安全のため簡易的にプロット
        plt.figure(figsize=(8, 6))
        sns.boxplot(x="Has_News", y="Volume", data=df, showfliers=False, palette="Set2")
        plt.title("ニュース有無と出来高(Volume)の関係")
        plt.xlabel("ニュースの有無")
        plt.ylabel("出来高 (Volume)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "analysis_news_impact_volume.png", dpi=300)
        plt.close()


# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    print(f"分析開始: {PROJECT_ROOT}")

    # 1-3. 統計CSVからの分析
    analyze_news_distribution()
    analyze_news_trend()
    analyze_sentiment_trend()

    # 4-6. 全データセットを用いた詳細分析
    analyze_dataset_stats()

    print(f"\n全分析完了。画像は以下に保存されました:\n{OUTPUT_DIR}")
