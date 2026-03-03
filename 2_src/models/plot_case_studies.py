from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定エリア
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_deep_strict"
PRED_CSV_PATH = DATA_DIR / "predictions_FusionTransformer.csv"

# 共通設定
SHIFT_DAYS = 7  # 予測線の左シフト量
WINDOW_DAYS = 60  # 表示期間（前後60日＝約4ヶ月分を表示）

# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def plot_refined_case(target_code, target_date_str, mode, event_label, news_sentiment, save_name):
    print(f"Plotting {target_code} ({mode}) for date {target_date_str}...")

    # 1. データ読み込み
    try:
        df = pd.read_csv(PRED_CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        print("ファイルが見つかりません")
        return

    # 2. データ抽出・加工
    df_code = df[df["code"] == target_code].sort_values("Date").reset_index(drop=True)
    df_code["Target_Date"] = df_code["Date"] + timedelta(days=5)
    df_code["Pred_Plot_Date"] = df_code["Target_Date"] - timedelta(days=SHIFT_DAYS)

    # 3. 期間フィルタリング
    center_date = pd.to_datetime(target_date_str)
    start_date = center_date - timedelta(days=WINDOW_DAYS)
    end_date = center_date + timedelta(days=WINDOW_DAYS)

    # データ範囲内でのクリップ
    min_date, max_date = df_code["Target_Date"].min(), df_code["Target_Date"].max()
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date

    plot_data = df_code[(df_code["Target_Date"] >= start_date) & (df_code["Target_Date"] <= end_date)].copy()

    if len(plot_data) == 0:
        print("データがありません")
        return

    # 4. プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # -----------------------------
    # 上段: 株価チャート
    # -----------------------------
    ax1.plot(
        plot_data["Target_Date"], plot_data["Actual"], color="#333333", label="Actual Price", linewidth=2.0, alpha=0.8
    )
    ax1.plot(
        plot_data["Pred_Plot_Date"],
        plot_data["Pred"],
        color="#d62728",
        linestyle="--",
        label="Prediction (AI)",
        linewidth=2.5,
    )

    # イベントライン
    ax1.axvline(center_date, color="green", linestyle=":", alpha=0.6, linewidth=1.5)

    # --- レイアウト調整 ---
    # Y軸の余白確保（吹き出し用）
    y_min = plot_data["Actual"].min()
    y_max = plot_data["Actual"].max()
    y_range = y_max - y_min
    ax1.set_ylim(y_min - y_range * 0.1, y_max + y_range * 0.4)  # 上に広めに

    # X軸の範囲（両端をしっかり表示）
    # データが存在する範囲できっちり切ることで、空白をなくす
    ax1.set_xlim(plot_data["Target_Date"].min(), plot_data["Target_Date"].max())

    # --- 吹き出し ---
    y_val = plot_data.loc[plot_data["Target_Date"] == center_date, "Actual"].mean()
    if not plot_data[plot_data["Target_Date"] == center_date].empty:
        y_val = plot_data.loc[plot_data["Target_Date"] == center_date, "Actual"].values[0]

    ax1.annotate(
        f"{event_label}\n(Sentiment: {news_sentiment})",
        xy=(center_date, y_val),
        xytext=(0.5, 0.95),
        textcoords="axes fraction",  # 中央上部固定
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=10),
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.9),
    )

    stock_name = plot_data["Name"].iloc[0]
    ax1.set_title(f"Case Study: {stock_name} ({target_code}) - {mode} Case", fontsize=16, pad=15)
    ax1.set_ylabel("Stock Price (JPY)", fontsize=14)
    ax1.legend(loc="lower left", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # -----------------------------
    # 下段: Gate Score
    # -----------------------------
    gate_val = 0
    if not plot_data[plot_data["Target_Date"] == center_date].empty:
        gate_val = plot_data.loc[plot_data["Target_Date"] == center_date, "Gate_Score"].values[0]

    ax2.plot(plot_data["Target_Date"], plot_data["Gate_Score"], color="#1f77b4", label="Gate Score", linewidth=2.5)
    ax2.fill_between(plot_data["Target_Date"], plot_data["Gate_Score"], 0, color="#1f77b4", alpha=0.2)

    # Gate状態
    status = "OPEN" if gate_val > 0.3 else "CLOSED"
    color = "red" if status == "OPEN" else "gray"

    ax2.scatter([center_date], [gate_val], color=color, s=100, zorder=5)

    ax2.annotate(
        f"Gate {status}\n(Score: {gate_val:.2f})",
        xy=(center_date, gate_val),
        xytext=(0.5, 0.9),
        textcoords="axes fraction",
        ha="center",
        va="top",
        fontsize=12,
        color=color,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=color, linewidth=2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9),
    )

    ax2.axvline(center_date, color="green", linestyle=":", alpha=0.6, linewidth=1.5)
    ax2.axhline(0.3, color="gray", linestyle="--", linewidth=1.0, label="Threshold")

    ax2.set_ylabel("Gate Score", fontsize=14)
    ax2.set_ylim(0, 0.7)
    ax2.legend(loc="upper right", fontsize=12)
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=0, fontsize=12)
    plt.tight_layout()

    plt.savefig(save_name, dpi=300)
    print(f"Saved: {save_name}")
    plt.close()


# ==========================================
# 実行
# ==========================================

# 1. 日本製鉄
plot_refined_case(
    target_code=5401,
    target_date_str="2025-06-17",
    mode="High Attention",
    event_label="Major News Event\n(Strategic Partnership etc.)",
    news_sentiment="Positive / High Impact",
    save_name="Refined_Case1_NipponSteel_V2.png",
)

# 2. ディスコ
plot_refined_case(
    target_code=6146,
    target_date_str="2025-01-20",
    mode="Noise Blocking",
    event_label="High Volatility\n(Regarded as Noise)",
    news_sentiment="Neutral / Low Relevance",
    save_name="Refined_Case2_Disco_V2.png",
)
