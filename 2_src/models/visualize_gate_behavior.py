from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(font="MS Gothic")

PROJECT_ROOT = Path(".").resolve()
TARGET_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
OUTPUT_DIR = TARGET_DIR

# ターゲットファイル名
TARGET_FILE = "predictions_MultiModalGatedTransformer.csv"


def plot_gate_dynamics(df, code):
    # データ抽出 (指定銘柄、直近150日)
    df_code = df[df["code"] == code].sort_values("Date")
    if len(df_code) > 150:
        df_code = df_code.iloc[-150:]

    if len(df_code) == 0:
        print(f"No data found for code {code}")
        return

    dates = df_code["Date"]
    price = df_code["Close"]

    # Gate Scoreがある場合のみプロット
    if "Gate_Score" not in df_code.columns:
        print("Gate_Score column not found.")
        return

    gate_score = df_code["Gate_Score"]

    fig, ax1 = plt.subplots(figsize=(14, 8))

    # 株価プロット (左軸)
    ax1.plot(dates, price, color="black", label="Stock Price", alpha=0.7, linewidth=1.2)
    ax1.set_ylabel("Stock Price")
    ax1.set_xlabel("Date")

    # Gate Scoreプロット (右軸)
    ax2 = ax1.twinx()
    # 塗りつぶしで表現
    ax2.fill_between(dates, gate_score, 0, color="orange", alpha=0.3, label="Gate Openness (Text Impact)")
    ax2.plot(dates, gate_score, color="darkorange", linewidth=1.0)
    ax2.set_ylabel("Gate Score (0.0 - 1.0)")
    ax2.set_ylim(0, 1.05)

    # タイトル
    name = df_code["Name"].iloc[0] if "Name" in df_code.columns else code
    plt.title(f"Gate Mechanism Analysis: {name} ({code})\n(Multi-Modal Gated Transformer)")

    # 凡例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = OUTPUT_DIR / f"MultiModalGatedTransformer_gate_analysis_{code}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")


def main():
    print("--- Starting Gate Analysis ---")
    file_path = TARGET_DIR / TARGET_FILE

    if not file_path.exists():
        print(f"Error: {TARGET_FILE} not found in {TARGET_DIR}")
        return

    print(f"Loading {file_path.name}...")
    df = pd.read_csv(file_path)
    df["Date"] = pd.to_datetime(df["Date"])

    # 代表的な銘柄についてプロット (例: 9984 ソフトバンクG, 7203 トヨタ)
    # データに含まれる銘柄コードを確認
    available_codes = df["code"].unique()

    # プロットしたい銘柄リスト
    target_codes = [9984, 7203, 6758]  # SBG, Toyota, Sony

    for code in target_codes:
        if code in available_codes:
            plot_gate_dynamics(df, code)
        else:
            # コードが整数か文字列かで一致しない場合があるので念のためキャストして確認
            if str(code) in df["code"].astype(str).values:
                plot_gate_dynamics(df, str(code))
            elif int(code) in df["code"].astype(int).values:
                plot_gate_dynamics(df, int(code))
            else:
                # データ内の最初の銘柄をプロット
                if code == target_codes[0]:
                    first_code = df["code"].iloc[0]
                    print(f"Code {code} not found. Plotting first available code: {first_code}")
                    plot_gate_dynamics(df, first_code)


if __name__ == "__main__":
    main()
