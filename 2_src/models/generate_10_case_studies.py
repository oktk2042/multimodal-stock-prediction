from datetime import timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
PROJECT_ROOT = Path(".").resolve()

# ■ 入力ファイル (Gate Score入り予測データ)
# アップロードされたファイル名に合わせています
DATA_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
REPO_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"
PRED_FILE = REPO_DIR / "predictions_MultiModalGatedTransformer.csv"

# ■ 事例リストファイル
CASE_POS_FILE = DATA_DIR / "case_study_positive.csv"
CASE_NEG_FILE = DATA_DIR / "case_study_negative_relaxed.csv"
CASE_CLOSED_FILE = DATA_DIR / "case_study_gate_closed_final.csv"

# ■ 出力先
OUTPUT_DIR = PROJECT_ROOT / "final_figures_10_cases_real"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ■ フォント設定
target_fonts = [
    "MS Gothic",
    "Hiragino Maru Gothic Pro",
    "Yu Gothic",
    "Meiryo",
    "TakaoGothic",
    "IPAGothic",
    "DejaVu Sans",
]
fonts = [f.name for f in fm.fontManager.ttflist]
found_font = next((f for f in target_fonts if f in fonts), "sans-serif")

plt.rcParams["font.family"] = found_font
plt.rcParams["font.size"] = 14
plt.rcParams["axes.linewidth"] = 1.5
sns.set(style="whitegrid", font=found_font)


# ==========================================
# 2. 描画関数 (論文用・最適化版)
# ==========================================
def plot_case_study(df, code, name, case_type, event_date, title, score, filename):
    """
    株価(左軸)とGateスコア(右軸)を描画。
    Gateスコアは実データを使用。
    """
    if df.empty:
        print(f"Skipping {code}: No data found.")
        return False

    # 日付ソートと範囲確認
    df = df.sort_values("Date")
    event_date = pd.to_datetime(event_date)

    # イベント日がデータ範囲外ならスキップ
    if not (df["Date"].min() <= event_date <= df["Date"].max()):
        print(f"Skipping {code}: Event date {event_date.date()} out of range.")
        return False

    fig, ax1 = plt.subplots(figsize=(12, 7))

    # --- 左軸: 株価 (Stock Price) ---
    color_price = "#2c3e50"  # Dark Blue/Gray
    ax1.plot(df["Date"], df["Close"], color=color_price, linewidth=2.5, label="Stock Price")
    ax1.set_ylabel("Stock Price (JPY)", color=color_price, fontsize=14, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_price)
    ax1.grid(True, linestyle="--", alpha=0.5)

    # --- 右軸: Gate Score ---
    ax2 = ax1.twinx()

    # ケースタイプによる色分け
    is_open = "Open" in case_type
    color_gate = "#e74c3c" if is_open else "#7f8c8d"  # Red for Open, Gray for Closed

    # 実データのGate Scoreをプロット
    if "Gate_Score" in df.columns:
        ax2.fill_between(df["Date"], df["Gate_Score"], 0, color=color_gate, alpha=0.2)
        ax2.plot(df["Date"], df["Gate_Score"], color=color_gate, linewidth=2.0, linestyle="-", label="Gate Score")
    else:
        # 万が一Gate列がない場合 (通常ありえないが安全策)
        print(f"Warning: No Gate_Score for {code}, skipping plot.")
        plt.close()
        return False

    # 軸設定
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.3, color="black", linestyle=":", linewidth=1.5, alpha=0.7, label="Threshold (0.3)")
    ax2.set_ylabel("Gate Score", color=color_gate, fontsize=14, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_gate)

    # --- 注釈 (Smart Annotation) ---
    # イベント当日のGate値を取得
    nearest_idx = (df["Date"] - event_date).abs().idxmin()
    gate_val = df.loc[nearest_idx, "Gate_Score"]

    # タイトル整形
    short_title = (str(title)[:15] + "...") if title and len(str(title)) > 15 else "Event"
    score_str = f"{score:.2f}" if score is not None else f"{gate_val:.2f}"

    anno_text = f"{case_type}\nScore: {score_str}\n{short_title}"

    # 吹き出し位置の自動調整 (Gateが高いなら下へ、低いなら上へ)
    if gate_val > 0.6:
        xytext = (event_date + timedelta(days=5), gate_val - 0.25)
        connectionstyle = "arc3,rad=0.2"
    else:
        xytext = (event_date + timedelta(days=5), gate_val + 0.25)
        connectionstyle = "arc3,rad=-0.2"

    # Y座標が枠外に出ないようにクリップ
    xytext = (xytext[0], np.clip(xytext[1], 0.1, 0.95))

    ax2.annotate(
        anno_text,
        xy=(event_date, gate_val),
        xytext=xytext,
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, connectionstyle=connectionstyle),
        fontsize=11,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color_gate, alpha=0.9, linewidth=2),
    )

    # 垂直線 (イベント発生日)
    ax1.axvline(event_date, color="green", linestyle="--", alpha=0.6, linewidth=1.5)

    # タイトルとレイアウト
    plt.title(f"Case Study: {name} ({code}) - {case_type}", fontsize=16, pad=15)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    fig.autofmt_xdate()

    plt.tight_layout()
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300)
    print(f"[Success] Saved: {save_path}")
    plt.close()
    return True


# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("データ読み込み中...")

    # 1. 予測結果ファイル (時系列データ) の読み込み
    try:
        # Codeを文字列として読み込む
        df_pred = pd.read_csv(PRED_FILE, dtype={"code": str})
        df_pred["Date"] = pd.to_datetime(df_pred["Date"])
        print(f"Prediction data loaded: {len(df_pred)} rows")

        # Gate Scoreが含まれているか確認
        if "Gate_Score" not in df_pred.columns:
            print("Error: 'Gate_Score' column is missing in predictions file.")
            return

    except FileNotFoundError:
        print(f"Error: {PRED_FILE} not found. Please check the file path.")
        return

    # 処理対象の定義
    # (ファイルパス, Gate状態ラベル, ファイル識別子)
    targets = [
        (CASE_POS_FILE, "Open (Positive)", "Positive"),
        (CASE_CLOSED_FILE, "Closed (Noise)", "Noise"),
        (CASE_NEG_FILE, "Open (Negative)", "Negative"),
    ]

    total_generated = 0

    for file_path, gate_label, label_id in targets:
        if not file_path.exists():
            print(f"Warning: {file_path} not found. Skipping {label_id} cases.")
            continue

        print(f"\n--- Processing {label_id} Cases ---")
        try:
            df_cases = pd.read_csv(file_path, dtype={"code": str, "Code": str})
            # カラム名統一
            if "Code" in df_cases.columns:
                df_cases.rename(columns={"Code": "code"}, inplace=True)

            count = 0
            # 上位から順に処理 (最大5件)
            for _, row in df_cases.iterrows():
                if count >= 5:
                    break

                code = str(row["code"])

                # 日付カラムの揺らぎ吸収
                date_col = "Date" if "Date" in df_cases.columns else "date"
                event_date = pd.to_datetime(row[date_col])

                name = row.get("Name", code)
                title = row.get("Title", "Market Event")

                # スコア取得 (ファイルによってカラム名が違う場合に対応)
                score = row.get("News_Sentiment", row.get("Gate_Score", 0.0))

                # 時系列データの抽出 (前後60日)
                mask = (
                    (df_pred["code"] == code)
                    & (df_pred["Date"] >= event_date - timedelta(days=60))
                    & (df_pred["Date"] <= event_date + timedelta(days=60))
                )
                df_plot = df_pred[mask].copy()

                # 描画実行
                if plot_case_study(
                    df_plot,
                    code,
                    name,
                    f"Gate {gate_label}",
                    event_date,
                    title,
                    score,
                    f"case_study_{code}_{label_id}.png",
                ):
                    count += 1
                    total_generated += 1

        except Exception as e:
            print(f"Error processing {label_id}: {e}")

    print(f"\n完了: 合計 {total_generated} 枚の画像を生成しました。")
    print(f"保存先: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
