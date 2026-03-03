from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 設定エリア
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
SAVE_NAME = "Backtest_All_Models_LS_Wide.png"

model_files = {
    "Proposed (Gated)": "predictions_MultiModalGatedTransformer.csv",
    "PatchTST": "predictions_PatchTST.csv",
    "Attention-LSTM": "predictions_AttentionLSTM.csv",
    "DLinear": "predictions_DLinear.csv",
    "iTransformer": "predictions_iTransformer.csv",
    "Vanilla Transformer": "predictions_VanillaTransformer.csv",
    "LightGBM": "predictions_LightGBM.csv",
    "Ridge Regression": "predictions_RidgeRegression.csv",
}

TOP_K = 5
GATE_THRESHOLD = 0.25

plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def calculate_ls_cumulative(file_path, model_name):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return None, None

    df["Date"] = pd.to_datetime(df["Date"])

    if "Daily_Return_Approx" not in df.columns:
        if "Actual" in df.columns and "Current" in df.columns:
            df["Return_5D"] = (df["Actual"] - df["Current"]) / df["Current"]
            df["Daily_Return_Approx"] = (1 + df["Return_5D"]) ** (1 / 5) - 1
        else:
            return None, None

    dates = sorted(df["Date"].unique())
    portfolio_returns = []
    valid_dates = []

    for d in dates:
        day_data = df[df["Date"] == d]

        if "Proposed" in model_name and "Gate_Score" in day_data.columns:
            target_data = day_data[day_data["Gate_Score"] > GATE_THRESHOLD]
        else:
            target_data = day_data

        if len(target_data) >= (TOP_K * 2):
            sorted_data = target_data.sort_values("Pred_Return", ascending=False)
            long_ret = sorted_data.head(TOP_K)["Daily_Return_Approx"].mean()
            short_ret = -1 * sorted_data.tail(TOP_K)["Daily_Return_Approx"].mean()
            port_ret = 0.5 * long_ret + 0.5 * short_ret
            portfolio_returns.append(port_ret)
            valid_dates.append(d)
        else:
            portfolio_returns.append(0.0)
            valid_dates.append(d)

    cumulative = (1 + np.array(portfolio_returns)).cumprod()
    total_ret = cumulative[-1] - 1
    std = np.std(portfolio_returns)
    sharpe = (np.mean(portfolio_returns) * 252) / (std * (252**0.5)) if std > 1e-9 else 0.0

    return pd.DataFrame({"Date": valid_dates, "Cumulative": cumulative}), (total_ret, sharpe)


def plot_all_models_grid():
    # GridSpecを使って領域を「3:1」に分割 (上:グラフ, 下:テーブル)
    fig = plt.figure(figsize=(15, 14))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.5)

    ax_chart = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis("off")  # テーブル領域の枠線を消す

    # カラーパレット
    palette = sns.color_palette("muted", n_colors=10)
    model_colors = {}
    c_idx = 0
    for m in model_files.keys():
        if "Proposed" in m:
            model_colors[m] = "#d62728"  # 赤
        else:
            if c_idx < len(palette) and palette[c_idx] == (0.839, 0.152, 0.156):
                c_idx += 1
            model_colors[m] = palette[c_idx % len(palette)]
            c_idx += 1

    # 市場平均プロット (1つのファイルから代表して取得)
    base_file = DATA_DIR / list(model_files.values())[0]
    if base_file.exists():
        base_df = pd.read_csv(base_file)
        base_df["Date"] = pd.to_datetime(base_df["Date"])
        base_df["Return_5D"] = (base_df["Actual"] - base_df["Current"]) / base_df["Current"]
        base_df["Daily_Return_Approx"] = (1 + base_df["Return_5D"]) ** (1 / 5) - 1
        market_returns = base_df.groupby("Date")["Daily_Return_Approx"].mean()
        market_cum = (1 + market_returns).cumprod()
        ax_chart.plot(
            market_cum.index,
            market_cum.values,
            label="Market Average",
            color="gray",
            linestyle="--",
            linewidth=2.0,
            alpha=0.7,
        )

    stats_data = []

    # 各モデルのプロット
    for model_name, file_name in model_files.items():
        res_df, metrics = calculate_ls_cumulative(DATA_DIR / file_name, model_name)
        if res_df is not None:
            lw = 4.5 if "Proposed" in model_name else 1.5
            alpha = 1.0 if "Proposed" in model_name else 0.6
            zorder = 10 if "Proposed" in model_name else 1

            ax_chart.plot(
                res_df["Date"],
                res_df["Cumulative"],
                label=model_name,
                color=model_colors[model_name],
                linewidth=lw,
                alpha=alpha,
                zorder=zorder,
            )

            stats_data.append({"Model": model_name, "Total Return": metrics[0], "Sharpe Ratio": metrics[1]})

    # --- グラフ設定 ---
    ax_chart.axhline(1.0, color="black", linewidth=1.0)
    ax_chart.set_title("Long-Short Strategy Performance Comparison (2025)", fontsize=20, fontweight="bold", pad=10)
    ax_chart.set_ylabel("Cumulative Return (Start=1.0)", fontsize=16)
    ax_chart.grid(True, alpha=0.3)
    ax_chart.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax_chart.tick_params(axis="both", which="major", labelsize=14)

    # 凡例をグラフエリアの下（テーブルの上）に配置
    ax_chart.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=15, frameon=False)

    # --- テーブル作成 ---
    stats_data.sort(key=lambda x: x["Total Return"], reverse=True)

    cell_text = []
    row_colors = []
    for rank, d in enumerate(stats_data, 1):
        cell_text.append([f"{rank}", d["Model"], f"{d['Total Return']:+.2%}", f"{d['Sharpe Ratio']:.2f}"])
        if "Proposed" in d["Model"]:
            row_colors.append("#ffe6e6")
        else:
            row_colors.append("white")

    table = ax_table.table(
        cellText=cell_text,
        colLabels=["Rank", "Model", "Total Return", "Sharpe Ratio"],
        colWidths=[0.05, 0.4, 0.15, 0.15],
        cellColours=[[c] * 4 for c in row_colors],
        loc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1, 1.8)

    # ヘッダー装飾
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    plt.savefig(SAVE_NAME, dpi=300, bbox_inches="tight", pad_inches=0.5)
    print(f"Saved: {SAVE_NAME}")


if __name__ == "__main__":
    plot_all_models_grid()
