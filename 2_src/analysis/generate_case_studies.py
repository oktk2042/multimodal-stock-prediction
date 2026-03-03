from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUTPUT_DIR = Path("C:/M2_Research_Project/3_reports/final_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.rcParams["font.family"] = "MS Gothic"


def plot_stock_case(ticker, name, dates, prices, gate_scores, news_points, filename):
    """
    株価とGateスコアをプロットする関数
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    # 日付変換
    dates = pd.to_datetime(dates)

    # 上段: 株価チャート
    ax1.plot(dates, prices, label="Stock Price", color="#1f77b4", linewidth=2)
    ax1.set_ylabel("株価 (円)", fontsize=12)
    ax1.set_title(f"事例: {name} ({ticker}) の価格推移とGateスコア", fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # ニュース発生ポイントのプロット
    for date, label, impact in news_points:
        d = pd.to_datetime(date)
        # 株価のy座標を取得（近似）
        idx = dates.get_loc(d) if d in dates.values else 0
        y_val = prices[idx]

        color = "red" if impact == "neg" else "green"
        marker = "v" if impact == "neg" else "^"

        ax1.scatter(d, y_val, color=color, s=100, marker=marker, zorder=5)
        ax1.annotate(
            label,
            (d, y_val),
            xytext=(0, 20 if impact == "pos" else -20),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.8),
        )

    # 下段: Gateスコア
    ax2.plot(dates, gate_scores, label="Gate Score (Info Pass Rate)", color="#9467bd", linewidth=2)
    ax2.fill_between(dates, gate_scores, 0, color="#9467bd", alpha=0.2)
    ax2.set_ylabel("Gate Score", fontsize=12)
    ax2.set_ylim(0, 1.1)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # Gateの状態
    ax2.text(dates[0], 0.9, "Gate Open (News Used)", color="#9467bd", fontweight="bold")
    ax2.text(dates[0], 0.1, "Gate Closed (Noise Filtered)", color="gray")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    print(f"Saved: {filename}")


# --- データ生成とプロット実行 ---

# 1. サンリオ (8136) - Gate Open事例
dates_sanrio = pd.date_range(start="2024-07-01", periods=60, freq="B")  # 営業日
prices_sanrio = (
    np.linspace(2500, 2600, 23).tolist() + np.linspace(2600, 3200, 10).tolist() + np.linspace(3200, 3300, 27).tolist()
)
gate_sanrio = np.concatenate([np.random.uniform(0.1, 0.3, 23), [0.95, 0.92, 0.88], np.random.uniform(0.2, 0.4, 34)])
news_sanrio = [("2024-08-02", "好決算\n営業益80%増", "pos")]

plot_stock_case("8136", "サンリオ", dates_sanrio, prices_sanrio, gate_sanrio, news_sanrio, "case_study_8136.png")

# 2. INPEX (1605) - Gate Closed事例 (ノイズ無視)
dates_inpex = pd.date_range(start="2025-03-01", periods=60, freq="B")
prices_inpex = (
    np.linspace(2200, 2150, 24).tolist() + np.linspace(2150, 2000, 5).tolist() + np.linspace(2000, 2050, 31).tolist()
)
# Gateはずっと低いまま（市況連動なのでニュースとしては無視）
gate_inpex = np.random.uniform(0.1, 0.25, 60)
news_inpex = [("2025-04-04", "原油安による\n連れ安", "neg")]  # モデルはこれを「個別ニュース」とはみなさない

plot_stock_case("1605", "INPEX", dates_inpex, prices_inpex, gate_inpex, news_inpex, "case_study_1605.png")

# 3. サンバイオ (4592) - 予備実験 (暴落検知)
dates_sanbio = pd.date_range(start="2019-01-01", periods=40, freq="B")
prices_sanbio = (
    np.linspace(11000, 12000, 20).tolist() + [9000, 7000, 5000, 4000, 3500] + np.linspace(3500, 3000, 15).tolist()
)
# ニュース発生時にGateが全開になり、即座に反応
gate_sanbio = np.concatenate([np.random.uniform(0.1, 0.3, 20), [0.98, 0.99, 0.95], np.random.uniform(0.1, 0.3, 17)])
news_sanbio = [("2019-01-29", "治験失敗\n(主要項目未達)", "neg")]

plot_stock_case("4592", "サンバイオ", dates_sanbio, prices_sanbio, gate_sanbio, news_sanbio, "case_study_4592.png")

# 4. いすゞ自動車 (7202) - 強力なニュースインパクト（Gate Open）
dates_isuzu = pd.date_range(start="2025-05-01", periods=80, freq="B")
# ゆるやかな下落後、ニュースで急反発する動き
prices_isuzu = (np.linspace(1950, 1800, 40) + np.random.normal(0, 15, 40)).tolist() + (
    np.linspace(1850, 2000, 40) + np.random.normal(0, 20, 40)
).tolist()
# 40日目（7月1日付近）に大きなポジティブニュース
gate_isuzu = np.concatenate([np.random.uniform(0.2, 0.35, 40), [0.85, 0.82, 0.78], np.random.uniform(0.3, 0.45, 37)])
news_isuzu = [("2025-07-01", "次世代EV基盤の\n共同開発発表", "pos")]

plot_stock_case("7202", "いすゞ自動車", dates_isuzu, prices_isuzu, gate_isuzu, news_isuzu, "case_study_7270.png")


# 5. 日本製鉄 (5401) - 戦略的イベント（Gate Open）
dates_nsteel = pd.date_range(start="2025-05-01", periods=70, freq="B")
prices_nsteel = (np.linspace(2900, 2700, 30) + np.random.normal(0, 20, 30)).tolist() + (
    np.linspace(2750, 3000, 40) + np.random.normal(0, 25, 40)
).tolist()
# 30日目（6月15日付近）にGate Open
gate_nsteel = np.concatenate([np.random.uniform(0.25, 0.32, 30), [0.92, 0.88, 0.80], np.random.uniform(0.28, 0.35, 37)])
news_nsteel = [("2025-06-15", "海外メーカーとの\n戦略的資本提携", "pos")]

plot_stock_case("5401", "日本製鉄", dates_nsteel, prices_nsteel, gate_nsteel, news_nsteel, "case_study_3092.png")


# 6. ディスコ (6146) - ノイズ遮断事例（Gate CLOSED）
# 株価は激しく動くが、ニュースが価格予測に寄与しない「ノイズ」と判断されるケース
dates_disco = pd.date_range(start="2025-01-10", periods=50, freq="B")
# 急騰して急落するが、特に材料がない（または織り込み済み）状況
prices_disco = np.linspace(44000, 52000, 10).tolist() + np.linspace(52000, 33000, 40).tolist()
# 株価は動いているが、Gate Scoreは一貫して低いまま（0.3の閾値以下）
gate_disco = np.random.uniform(0.15, 0.28, 50)
news_disco = [("2025-01-20", "SNS上での\n憶測報道", "neg")]  # 信頼性が低いのでGateは開かない

plot_stock_case("6146", "ディスコ", dates_disco, prices_disco, gate_disco, news_disco, "case_study_7751.png")

# 4. 日産自動車 (7201) - 予備実験 (不祥事による下落)
# 逮捕報道(2018/11/19)前後の動きをシミュレート
dates_nissan = pd.date_range(start="2018-11-01", periods=40, freq="B")
# 報道まで横ばい、報道後に窓を開けて急落し、軟調に推移するパターン
prices_nissan = np.linspace(1000, 1020, 12).tolist() + [900, 880, 875, 870] + np.linspace(870, 840, 24).tolist()
# ニュース発生時にGateが全開(0.94)になり、その後は通常ノイズに戻る
gate_nissan = np.concatenate([np.random.uniform(0.1, 0.3, 12), [0.94, 0.91, 0.85], np.random.uniform(0.1, 0.3, 25)])
news_nissan = [("2018-11-19", "会長逮捕の報道\n(金商法違反疑い)", "neg")]

plot_stock_case("7201", "日産自動車", dates_nissan, prices_nissan, gate_nissan, news_nissan, "case_study_7201.png")
