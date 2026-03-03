from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定 & 日本語フォント
# ==========================================
matplotlib.rcParams["font.family"] = "MS Gothic"  # Windowsの場合
# matplotlib.rcParams['font.family'] = 'Hiragino Sans' # Macの場合
sns.set(font="MS Gothic", style="whitegrid")

# パス設定
PROJECT_ROOT = Path("C:/M2_Research_Project")
INPUT_FILE = PROJECT_ROOT / "3_reports" / "phase3_production_strict" / "detailed_analysis_metrics.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "final_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_sector_comparison():
    print("セクター別精度比較グラフを作成中...")

    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} が見つかりません。")
        return

    # 1. データ読み込み
    df = pd.read_csv(INPUT_FILE)

    # 2. 必要なデータの抽出
    # Type='Sector' かつ 比較したいモデルのみ抽出
    # モデル名はCSVの中身に合わせて調整（ここでは 'FusionTransformer' と 'PatchTST' を想定）
    target_models = ["FusionTransformer", "PatchTST", "Attention-LSTM"]

    # モデル名が微妙に違う場合があるので、キーワードで抽出
    # "Fusion" や "Ours" を含むものを提案手法とする
    df["Model_Short"] = df["Model"].apply(lambda x: "Proposed" if "Fusion" in x or "Ours" in x else x)

    # フィルタリング
    df_sector = df[
        (df["Type"] == "Sector") & (df["Model_Short"].isin(["Proposed", "PatchTST", "LSTM"]))  # 比較対象
    ].copy()

    if df_sector.empty:
        print("指定したモデルまたはセクターデータが見つかりません。CSVの中身を確認してください。")
        print(df["Model"].unique())
        return

    # 3. 可視化
    plt.figure(figsize=(14, 8))

    # 棒グラフ (Hue=Model)
    chart = sns.barplot(
        data=df_sector,
        x="Category",  # セクター名
        y="Accuracy",
        hue="Model_Short",
        palette="viridis",  # 色使い（Proposedを目立たせるならカスタムパレット推奨）
    )

    # 4. デザイン調整
    plt.title("セクター別 方向正解率 (Accuracy) の比較", fontsize=16)
    plt.xlabel("セクター (Sector)", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.ylim(0.4, 0.7)  # 見やすくするためにY軸を調整 (0.5付近が攻防ラインのため)
    plt.axhline(0.5, color="red", linestyle="--", linewidth=1, label="Random (50%)")
    plt.legend(title="Model", loc="upper right")
    plt.xticks(rotation=45, ha="right")

    # 差分が大きいセクター（Energy, Basic Materialsなど）に注釈を入れるとベスト

    plt.tight_layout()
    save_path = OUTPUT_DIR / "final_sector_accuracy_comparison.png"
    plt.savefig(save_path, dpi=300)
    print(f"保存完了: {save_path}")


if __name__ == "__main__":
    plot_sector_comparison()
