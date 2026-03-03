from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
PRED_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
PRED_FILE = PRED_DIR / "predictions_MultiModalGatedTransformer.csv"
NEWS_FILE = DATA_DIR / "collected_news_historical_full.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports/analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# グラフ設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")

# 調査対象の6銘柄
TARGET_STOCKS = {
    "7203": "Toyota (High)",
    "7267": "Honda (High)",
    "6305": "Hitachi CM (Medium)",
    "6988": "Nitto Denko (Medium)",
    "2801": "Kikkoman (Low)",
    "2432": "DeNA (Low)",
}


def visualize_behavior_combined():
    # 1. データの読み込み
    print("予測結果とニュースデータを読み込み中...")
    df_pred = pd.read_csv(PRED_FILE)
    df_pred["code"] = df_pred["code"].astype(str).str.extract(r"(\d{4})")[0]
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])

    df_news = pd.read_csv(NEWS_FILE, low_memory=False)
    date_col = "Date" if "Date" in df_news.columns else "published"
    df_news[date_col] = pd.to_datetime(df_news[date_col], errors="coerce")
    df_news["str_code"] = df_news["Code"].astype(str).str.extract(r"(\d{4})")[0]

    # 2. 銘柄ごとにプロット作成
    for code, name in TARGET_STOCKS.items():
        print(f"銘柄 {name} を分析中...")

        # 銘柄データの抽出
        p_sub = df_pred[df_pred["code"] == code].sort_values("Date")
        n_sub = df_news[df_news["str_code"] == code]

        if p_sub.empty:
            print(f"Warning: No prediction data for {code}")
            continue

        # ニュースのデイリー集計
        daily_news = n_sub.resample("D", on=date_col).size().reindex(p_sub["Date"], fill_value=0)

        # --- プロット開始 (3段構成) ---
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, 1, figsize=(14, 12), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1]}
        )
        plt.subplots_adjust(hspace=0.2)

        # --- (1) 株価と予測値 (1週間=7日 左にずらす) ---
        p_sub["Predicted_Price"] = p_sub["Current"] * (1 + p_sub["Pred_Return"])

        # 実績値はそのままのDate
        ax1.plot(p_sub["Date"], p_sub["Actual"], label="Actual Price (T+5)", color="black", linewidth=1.5)

        # 予測値のX軸を 7日間 差し引いて（左にずらして）プロット
        shifted_date = p_sub["Date"] - pd.Timedelta(days=7)
        ax1.plot(
            shifted_date,
            p_sub["Predicted_Price"],
            label="Predicted Price (Shifted Left 1wk)",
            color="red",
            linestyle="--",
            alpha=0.7,
        )

        ax1.set_title(
            f"Behavior Analysis: {name}\n(Prediction Line Shifted Left by 1 Week)", fontsize=16, fontweight="bold"
        )
        ax1.set_ylabel("Stock Price")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # (2) Gateスコア (0: 数値重視, 1: テキスト重視)
        ax2.fill_between(p_sub["Date"], 0, p_sub["Gate_Score"], color="teal", alpha=0.3, label="Gate Score")
        ax2.plot(p_sub["Date"], p_sub["Gate_Score"], color="teal", linewidth=1)
        ax2.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
        ax2.set_ylabel("Gate Score")
        ax2.set_ylim(0.15, 0.45)
        ax2.legend(loc="upper left")
        ax2.grid(True, alpha=0.3)

        # (3) ニュース頻度 (棒グラフ)
        ax3.bar(daily_news.index, daily_news.values, color="orange", alpha=0.7, label="Daily News Count")
        ax3.set_ylabel("News Count")
        ax3.legend(loc="upper left")
        ax3.grid(True, alpha=0.3)

        # X軸設定
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.xticks(rotation=45)

        # 保存
        save_path = OUTPUT_DIR / f"behavior_analysis_{code}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {save_path}")


if __name__ == "__main__":
    visualize_behavior_combined()
