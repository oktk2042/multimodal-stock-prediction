import io
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定エリア
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
REAL_CSV_PATH = DATA_DIR / "news_sentiment_historical.csv"


# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def plot_sentiment_distribution(csv_path, save_name="news_sentiment_dist.png"):
    # 1. データ読み込み
    try:
        df = pd.read_csv(csv_path)
        print(f"実データを読み込みました: {len(df)}件")
    except FileNotFoundError:
        print("ファイルが見つかりません。サンプルデータで描画します。")
        # サンプルデータ（共有いただいたデータの一部）
        csv_data = """Date,Code,Title,News_Sentiment,Keyword,Source
2018/5/30,9999,伊政治混迷、市場揺らす,-0.417329,日経平均,GoogleNews
2018/5/30,9999,欧州政治不安で円高進行,-0.021285,円相場,GoogleNews
2018/5/30,2267,ヤクルトライトを発売,0.526015,ヤクルト,GoogleNews
2018/5/30,3086,総合スーパー4月既存店前年割れ,-0.531077,Ｊフロント,GoogleNews
2018/5/30,1925,キングスカイフロントまちびらき,0.571634,ハウス,GoogleNews
2018/5/30,2497,マンU、神童獲得に向け交渉開始,-0.349720,ユナイテッド,GoogleNews
2018/5/30,2802,味の素西井社長は毎朝階段で,0.449211,味の素,GoogleNews"""
        df = pd.read_csv(io.StringIO(csv_data))

    # 2. ヒストグラム作成
    plt.figure(figsize=(10, 6))

    # ヒストグラム
    sns.histplot(df["News_Sentiment"], bins=30, kde=True, color="teal", alpha=0.6, edgecolor="white")

    # 統計情報のライン
    mean_val = df["News_Sentiment"].mean()
    plt.axvline(mean_val, color="red", linestyle="--", linewidth=1.5, label=f"Mean: {mean_val:.3f}")
    plt.axvline(0, color="gray", linestyle="-", linewidth=0.8, alpha=0.5)

    # タイトルとラベル
    plt.title("Distribution of News Sentiment Scores (FinBERT)", fontsize=14)
    plt.xlabel("Sentiment Score (-1.0: Negative <---> +1.0: Positive)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 注釈
    plt.text(
        0.5,
        plt.ylim()[1] * 0.85,
        "Positive News\n(New Products, Good Earnings)",
        color="green",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )
    plt.text(
        -0.5,
        plt.ylim()[1] * 0.85,
        "Negative News\n(Scandals, Market Crash)",
        color="red",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    # 3. 保存
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    print(f"[Done] 保存完了: {save_name}")


# --- 実行 ---
plot_sentiment_distribution(REAL_CSV_PATH)
