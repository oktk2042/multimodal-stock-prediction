from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定エリア (日本製鉄 5401 の事例)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_deep_strict"
PRED_CSV_PATH = DATA_DIR / "predictions_FusionTransformer.csv"

TARGET_CODE = 5401  # 日本製鉄
TARGET_DATE = "2025-09-22"  # Gateが開いた日
EVENT_LABEL = "Major News Event\n(M&A / Earnings)"  # ラベル
NEWS_SENTIMENT = "Positive / High Impact"

SHIFT_DAYS = 7  # 予測線の左シフト量
WINDOW_DAYS = 45  # 表示期間

# 日本語フォント
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def plot_open_case():
    # 1. データ読み込み
    try:
        df = pd.read_csv(PRED_CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
    except:
        print("ファイルが見つかりません")
        return

    # 2. データ抽出・加工
    df_code = df[df["code"] == TARGET_CODE].sort_values("Date").reset_index(drop=True)
    df_code["Target_Date"] = df_code["Date"] + timedelta(days=5)
    df_code["Pred_Plot_Date"] = df_code["Target_Date"] - timedelta(days=SHIFT_DAYS)

    # 3. 期間フィルタリング
    center_date = pd.to_datetime(TARGET_DATE)
    start_date = center_date - timedelta(days=WINDOW_DAYS)
    end_date = center_date + timedelta(days=WINDOW_DAYS)

    # スマートクリップ
    min_date, max_date = df_code["Target_Date"].min(), df_code["Target_Date"].max()
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date

    plot_data = df_code[(df_code["Target_Date"] >= start_date) & (df_code["Target_Date"] <= end_date)].copy()

    # 4. プロット
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # --- 上段: 株価 ---
    ax1.plot(
        plot_data["Target_Date"], plot_data["Actual"], color="#333333", label="Actual Price", linewidth=1.5, alpha=0.8
    )
    ax1.plot(
        plot_data["Pred_Plot_Date"],
        plot_data["Pred"],
        color="#d62728",
        linestyle="--",
        label="Prediction (AI)",
        linewidth=2.0,
    )

    # 注釈
    y_val = plot_data.loc[plot_data["Target_Date"] == center_date, "Actual"].mean()
    if not plot_data[plot_data["Target_Date"] == center_date].empty:
        y_val = plot_data.loc[plot_data["Target_Date"] == center_date, "Actual"].values[0]

    ax1.annotate(
        f"{EVENT_LABEL}\n(Sentiment: {NEWS_SENTIMENT})",
        xy=(center_date, y_val),
        xytext=(center_date + timedelta(days=5), y_val * 1.05),
        arrowprops=dict(facecolor="black", shrink=0.05),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
    )
    ax1.axvline(center_date, color="green", linestyle=":", alpha=0.6)

    ax1.set_title("Case Study: NIPPON STEEL (5401) - High Attention Response", fontsize=14)
    ax1.set_ylabel("Stock Price (JPY)")
    ax1.legend(loc="upper left")
    ax1.set_xlim(plot_data["Target_Date"].min(), plot_data["Target_Date"].max())
    ax1.grid(True, alpha=0.3)

    # --- 下段: Gate Score ---
    gate_val = 0
    if not plot_data[plot_data["Target_Date"] == center_date].empty:
        gate_val = plot_data.loc[plot_data["Target_Date"] == center_date, "Gate_Score"].values[0]

    ax2.plot(plot_data["Target_Date"], plot_data["Gate_Score"], color="#1f77b4", label="Gate Score", linewidth=2)
    ax2.fill_between(plot_data["Target_Date"], plot_data["Gate_Score"], 0, color="#1f77b4", alpha=0.2)

    # Gate Open強調
    ax2.scatter([center_date], [gate_val], color="red", s=50, zorder=5)
    ax2.annotate(
        f"Gate OPEN\n(Score: {gate_val:.2f})",
        xy=(center_date, gate_val),
        xytext=(center_date + timedelta(days=2), gate_val + 0.05),
        color="red",
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="red"),
    )

    ax2.axvline(center_date, color="green", linestyle=":", alpha=0.6)
    ax2.axhline(0.3, color="gray", linestyle="--", linewidth=0.8, label="Threshold")

    ax2.set_ylabel("Gate Score")
    ax2.set_ylim(0, 0.6)
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    # 日付整形
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    save_name = "Case1_NipponSteel_Open_Final.png"
    plt.savefig(save_name, dpi=300)
    print(f"保存しました: {save_name}")


if __name__ == "__main__":
    plot_open_case()
