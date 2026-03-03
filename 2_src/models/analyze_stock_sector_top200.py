from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(".").resolve()  # 実行環境に合わせて適宜調整してください
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# セクター情報ファイル (全銘柄)
SECTOR_FILE = DATA_DIR / "stock_sector_info.csv"
# 200銘柄のモデル用データセット
TARGET_DATASET = DATA_DIR / "dataset_for_modeling_top200_final.csv"

# 保存先
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_FILE = OUTPUT_DIR / "dataset_sector_distribution.png"


def plot_sector_distribution():
    # 1. 200銘柄のリストを取得
    if not TARGET_DATASET.exists():
        print(f"Error: Target dataset not found at {TARGET_DATASET}")
        return

    print(f"Loading target list from: {TARGET_DATASET.name}")
    df_prices = pd.read_csv(TARGET_DATASET, dtype={"Code": str, "code": str}, usecols=lambda c: c.lower() == "code")

    # カラム名統一 (Code or code)
    code_col = df_prices.columns[0]  # 読み込んだ唯一の列を使う
    top_200_codes = df_prices[code_col].unique()
    print(f"-> Target Stocks Found: {len(top_200_codes)}")

    # 2. セクター情報の読み込み
    if not SECTOR_FILE.exists():
        print(f"Error: Sector file not found at {SECTOR_FILE}")
        return

    print(f"Loading sector info from: {SECTOR_FILE.name}")
    df_sector = pd.read_csv(SECTOR_FILE, dtype={"Code": str, "code": str})

    # セクターファイル側のコード列名特定
    sec_code_col = "Code" if "Code" in df_sector.columns else "code"

    # 3. 200銘柄にフィルタリング
    df_target = df_sector[df_sector[sec_code_col].isin(top_200_codes)].copy()

    print(f"-> Matched Sector Info: {len(df_target)} / {len(top_200_codes)}")

    if len(df_target) == 0:
        print("Error: No matching stocks found in sector info.")
        return

    # 4. 可視化
    sector_counts = df_target["Sector"].value_counts()

    plt.figure(figsize=(12, 8))
    plt.rcParams["font.family"] = "MS Gothic"  # 日本語フォント
    plt.rcParams["font.size"] = 12

    # カラーパレット作成
    colors = sns.color_palette("pastel")[0 : len(sector_counts)]

    # 円グラフ
    wedges, texts, autotexts = plt.pie(
        sector_counts,
        labels=sector_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.5),  # ドーナツ型
    )

    plt.setp(texts, size=10)
    plt.setp(autotexts, size=9, weight="bold")

    plt.title(f"Sector Distribution of Target {len(top_200_codes)} Stocks", fontsize=16)
    plt.tight_layout()

    # 保存
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300)
    print(f"Saved plot to: {OUTPUT_FILE}")


if __name__ == "__main__":
    plot_sector_distribution()
