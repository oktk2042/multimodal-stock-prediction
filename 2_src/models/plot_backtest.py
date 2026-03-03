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
SAVE_NAME = "Backtest_Cumulative_Return_Gated.png"

# 戦略設定
STRATEGY_TYPE = "Gated Long-Only"
TOP_K = 5
GATE_THRESHOLD = 0.25

# 日本語フォント設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def calculate_metrics(daily_returns):
    """評価指標の計算"""
    cumulative = (1 + daily_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1

    n_days = len(daily_returns)
    sharpe = (daily_returns.mean() * 252) / (daily_returns.std() * (252**0.5) + 1e-6)

    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return total_return, sharpe, max_drawdown


def plot_backtest_gated():
    # 1. データ読み込み
    try:
        df = pd.read_csv(PRED_CSV_PATH)
        df["Date"] = pd.to_datetime(df["Date"])
    except Exception:
        print("ファイルが見つかりません")
        return

    # リターン計算
    df["Return_5D"] = (df["Actual"] - df["Current"]) / df["Current"]
    df["Daily_Return_Approx"] = (1 + df["Return_5D"]) ** (1 / 5) - 1

    dates = sorted(df["Date"].unique())

    portfolio_returns = []
    market_returns = []

    print(f"シミュレーション開始... ({len(dates)} 営業日)")

    for d in dates:
        day_data = df[df["Date"] == d]
        market_ret = day_data["Daily_Return_Approx"].mean()
        market_returns.append(market_ret)

        # Gated戦略
        high_attention_data = day_data[day_data["Gate_Score"] > GATE_THRESHOLD]
        if len(high_attention_data) > 0:
            top_stocks = high_attention_data.sort_values("Pred_Return", ascending=False).head(TOP_K)
            port_ret = top_stocks["Daily_Return_Approx"].mean()
        else:
            port_ret = 0.0
        portfolio_returns.append(port_ret)

    df_result = pd.DataFrame({"Date": dates, "Portfolio": portfolio_returns, "Market": market_returns})

    df_result["Cum_Portfolio"] = (1 + df_result["Portfolio"]).cumprod()
    df_result["Cum_Market"] = (1 + df_result["Market"]).cumprod()

    p_total, p_sharpe, p_mdd = calculate_metrics(df_result["Portfolio"])
    m_total, m_sharpe, m_mdd = calculate_metrics(df_result["Market"])

    # --- プロット開始 ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # メイン曲線をさらに太く (linewidth=4.0)
    ax.plot(
        df_result["Date"],
        df_result["Cum_Portfolio"],
        label=f"Proposed (Gate > {GATE_THRESHOLD})",
        color="#d62728",
        linewidth=4.0,
    )

    # 市場平均を太く (linewidth=2.5)
    ax.plot(
        df_result["Date"],
        df_result["Cum_Market"],
        label="Market Average (Buy & Hold)",
        color="gray",
        linestyle="--",
        linewidth=2.5,
    )

    ax.axhline(1.0, color="black", linewidth=1.0)

    # 文字サイズの底上げ
    ax.set_title("Backtest: Gated Strategy vs Market", fontsize=24, pad=25, fontweight="bold")
    ax.set_ylabel("Cumulative Return (Start=1.0)", fontsize=20)
    ax.set_xlabel("Date", fontsize=20)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True, alpha=0.3, linestyle=":")

    # 【修正】凡例を右下に配置し、サイズを20ptに拡大
    ax.legend(loc="lower right", fontsize=24, frameon=True, shadow=True, facecolor="white")

    # 【修正】統計情報を左上に配置し、フォントサイズを18ptに拡大。不透明度を1.0に。
    text_str = (
        f"Proposed (Gated):\n"
        f"  Total Return: {p_total:+.1%}\n"
        f"  Sharpe Ratio: {p_sharpe:.2f}\n\n"
        f"Market Avg:\n"
        f"  Total Return: {m_total:+.1%}\n"
        f"  Sharpe Ratio: {m_sharpe:.2f}"
    )
    # ボックスの設定: alpha=1.0 で背景を完全に白くし、文字を浮かせる
    props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=1.0, edgecolor="gray")

    # 座標 (0.02, 0.98) は左上の端
    ax.text(
        0.02,
        0.98,
        text_str,
        transform=ax.transAxes,
        fontsize=22,
        verticalalignment="top",
        bbox=props,
        family="monospace",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    plt.savefig(SAVE_NAME, dpi=300)
    print(f"Saved: {SAVE_NAME}")


if __name__ == "__main__":
    plot_backtest_gated()
