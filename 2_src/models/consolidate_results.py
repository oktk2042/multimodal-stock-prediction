import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 1. 環境設定 (ディレクトリパス)
# ==========================================

# プロジェクトルート (実行場所に応じて変更してください)
PROJECT_ROOT = Path(".").resolve()

# 入力元ディレクトリ (探索対象)
SOURCE_DIRS = [
    PROJECT_ROOT / "3_reports" / "phase3_production_deep_strict",
    PROJECT_ROOT / "3_reports" / "phase3_production_ridge_strict",
    PROJECT_ROOT / "3_reports" / "phase3_production_lgbm_strict",
]

# 出力先ディレクトリ (集約用)
DEST_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"

# 描画用フォント設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(font="MS Gothic")

# ==========================================
# 2. 変換ルール (Mapping)
# ==========================================
# Key: 元のファイル名やCSV内の古い表記に含まれるキーワード
# new_file: 新しいファイル名 (スペースなし)
# formal_name: 正式名称 (論文掲載用)
MODEL_MAP = {
    "FusionTransformer": {
        "new_file": "MultiModalGatedTransformer",
        "formal_name": "Multi-Modal Gated Transformer (Ours)",
    },
    "LSTM": {"new_file": "AttentionLSTM", "formal_name": "Attention-LSTM"},
    "Transformer": {  # Vanilla Transformer
        "new_file": "VanillaTransformer",
        "formal_name": "Vanilla Transformer",
    },
    "Ridge": {"new_file": "RidgeRegression", "formal_name": "Ridge Regression"},
    "LightGBM": {"new_file": "LightGBM", "formal_name": "LightGBM"},
    "DLinear": {"new_file": "DLinear", "formal_name": "DLinear"},
    "PatchTST": {"new_file": "PatchTST", "formal_name": "PatchTST"},
    "iTransformer": {"new_file": "iTransformer", "formal_name": "iTransformer"},
}

# 処理順序 (部分一致の誤爆を防ぐため、長い名前を先に)
PROCESS_ORDER = ["FusionTransformer", "iTransformer", "PatchTST", "LSTM", "DLinear", "LightGBM", "Ridge", "Transformer"]

# ==========================================
# 3. ユーティリティ関数
# ==========================================


def find_original_file(keyword, prefix, suffix, exclude_keywords=[]):
    """ソースディレクトリから条件に合うファイルを探す"""

    # 【修正箇所】prefixが空の場合に '**' が生成されないように分岐処理を追加
    if prefix:
        search_pattern = f"*{prefix}*{keyword}*{suffix}"
    else:
        # prefixがない場合は先頭の * は1つだけにする
        search_pattern = f"*{keyword}*{suffix}"

    for d in SOURCE_DIRS:
        if not d.exists():
            continue
        try:
            candidates = list(d.glob(search_pattern))
            for f in candidates:
                # 除外キーワードが含まれていないか確認
                if not any(ex in f.name for ex in exclude_keywords):
                    return f
        except ValueError as e:
            print(f"    [Pattern Error] {search_pattern} in {d}: {e}")
            continue

    return None


def update_model_name_in_df(df, target_col="Model"):
    """DataFrame内のモデル名を正式名称に変換する"""
    if target_col not in df.columns:
        return df

    # マッピング辞書作成
    name_map = {}
    for key, val in MODEL_MAP.items():
        name_map[key] = val["formal_name"]
        name_map[val["formal_name"]] = val["formal_name"]

    # 置換実行
    df[target_col] = df[target_col].replace(name_map)
    return df


# ==========================================
# 4. 描画ロジック
# ==========================================


def generate_plots(csv_path, formal_name, file_base, out_dir):
    """散布図、時系列予測(全体・拡大)の生成"""
    try:
        df = pd.read_csv(csv_path)

        # 日付変換
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])

        # 必須カラムチェック
        required_cols = ["Actual", "Current", "Pred_Return", "Pred", "code", "Date"]
        if not all(col in df.columns for col in required_cols):
            print(f"    [Skip Plot] Missing columns in {csv_path.name}")
            return

        # リターン計算
        actual_ret = (df["Actual"] - df["Current"]) / df["Current"]
        pred_ret = df["Pred_Return"]

        # 1. 散布図 (Scatter)
        plt.figure(figsize=(6, 6))
        plt.scatter(actual_ret, pred_ret, alpha=0.3, s=15, color="#1f77b4")
        min_val = min(actual_ret.min(), pred_ret.min())
        max_val = max(actual_ret.max(), pred_ret.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="Ideal")
        plt.axhline(0, color="gray", linestyle=":", linewidth=0.8)
        plt.axvline(0, color="gray", linestyle=":", linewidth=0.8)
        plt.title(f"{formal_name}\nReturn Scatter Plot")
        plt.xlabel("Actual Return")
        plt.ylabel("Predicted Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_dir / f"{file_base}_scatter.png", dpi=300)
        plt.close()

        # 2. 時系列予測 (代表銘柄: 9984 or 最初に見つかったもの)
        target_code = 9984
        if target_code not in df["code"].values:
            target_code = df["code"].unique()[0]

        df_plot = df[df["code"] == target_code].sort_values("Date").copy()

        # 全体プロット
        plt.figure(figsize=(12, 6))
        plt.plot(df_plot["Date"], df_plot["Actual"], label="Actual", color="black", linewidth=0.8, alpha=0.8)
        plt.plot(df_plot["Date"], df_plot["Pred"], label="Prediction", color="#d62728", linewidth=0.8, alpha=0.8)
        plt.title(f"{formal_name}: Prediction (Code: {target_code})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"{file_base}_pred_full.png", dpi=300)
        plt.close()

        # 拡大プロット (ラスト100日)
        df_zoom = df_plot.iloc[-100:]
        plt.figure(figsize=(12, 6))
        plt.plot(
            df_zoom["Date"],
            df_zoom["Actual"],
            label="Actual",
            color="black",
            marker=".",
            markersize=4,
            linestyle="-",
            linewidth=1,
        )
        plt.plot(
            df_zoom["Date"],
            df_zoom["Pred"],
            label="Prediction",
            color="#d62728",
            marker=".",
            markersize=4,
            linestyle="-",
            linewidth=1,
        )
        plt.title(f"{formal_name}: Prediction Zoom (Last 100 Days)")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(out_dir / f"{file_base}_pred_zoom.png", dpi=300)
        plt.close()

        print(f"    [Generated Plots] {file_base}")

    except Exception as e:
        print(f"    [Error Plotting] {e}")


def generate_importance_plot(csv_path, formal_name, file_base, out_dir):
    """特徴量重要度の棒グラフ生成"""
    try:
        df = pd.read_csv(csv_path)

        # カラム名の正規化
        col_map = {"gain": "Importance", "Value": "Importance", "importance": "Importance", "feature": "Feature"}
        df = df.rename(columns=col_map)

        if "Importance" in df.columns and "Feature" in df.columns:
            df = df.sort_values("Importance", ascending=False).head(20)

            plt.figure(figsize=(10, 8))
            plt.barh(df["Feature"][::-1], df["Importance"][::-1], color="teal")
            plt.title(f"{formal_name}\nFeature Importance (Top 20)")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(out_dir / f"{file_base}_feature_importance.png", dpi=300)
            plt.close()
            print("    [Generated Plot] Feature Importance")
        else:
            print(f"    [Skip Imp Plot] Columns not found: {list(df.columns)}")

    except Exception as e:
        print(f"    [Error Imp Plot] {e}")


# ==========================================
# 5. メイン処理
# ==========================================


def main():
    print("=== Start: Consolidate, Rename, Update & Regenerate ===")
    print(f"Output Directory: {DEST_DIR}")

    if not DEST_DIR.exists():
        DEST_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------
    # 1. 各モデルの個別ファイル処理
    # ------------------------------------------------
    for key in PROCESS_ORDER:
        config = MODEL_MAP[key]
        new_base = config["new_file"]
        formal_name = config["formal_name"]

        print(f"\n[{key}] Processing -> {formal_name}")

        # Transformer除外設定
        exclusions = []
        if key == "Transformer":
            exclusions = ["Fusion", "iTransformer"]

        # --- (A) Predictions CSV ---
        orig_pred = find_original_file(key, "predictions_", ".csv", exclusions)
        if orig_pred:
            new_pred_name = f"predictions_{new_base}.csv"
            dest_pred = DEST_DIR / new_pred_name
            shutil.copy2(orig_pred, dest_pred)

            # 画像生成
            generate_plots(dest_pred, formal_name, new_base, DEST_DIR)
        else:
            print("    [Warning] Prediction CSV not found.")

        # --- (B) Feature Importance CSV ---
        # prefixが空文字のケース（これがエラーの原因だった箇所）
        orig_imp = find_original_file(key, "", "feature_importance.csv", exclusions)
        if orig_imp:
            new_imp_name = f"{new_base}_feature_importance.csv"
            dest_imp = DEST_DIR / new_imp_name
            shutil.copy2(orig_imp, dest_imp)

            # 画像生成
            generate_importance_plot(dest_imp, formal_name, new_base, DEST_DIR)

        # --- (C) Learning Curve (Image Copy) ---
        orig_lc = find_original_file(key, "", "learning_curve.png", exclusions)
        if orig_lc:
            shutil.copy2(orig_lc, DEST_DIR / f"{new_base}_learning_curve.png")

        # --- (D) Best Model (.pth) ---
        orig_pth = find_original_file(key, "best_model_", ".pth", exclusions)
        if orig_pth:
            shutil.copy2(orig_pth, DEST_DIR / f"best_model_{new_base}.pth")

    # ------------------------------------------------
    # 2. サマリファイルの処理 (モデル名更新)
    # ------------------------------------------------
    print("\n[Processing Summary Files]")

    # --- model_comparison_summary.csv ---
    summary_file = find_original_file("model_comparison_summary", "", ".csv", [])
    if summary_file:
        dest_sum = DEST_DIR / "model_comparison_summary.csv"
        shutil.copy2(summary_file, dest_sum)
        try:
            df = pd.read_csv(dest_sum)
            df = update_model_name_in_df(df, "Model")
            df.to_csv(dest_sum, index=False)
            print("    [Updated] model_comparison_summary.csv (Model names updated)")
        except Exception as e:
            print(f"    [Error] Updating summary: {e}")

    # --- final_model_comparison.csv ---
    final_file = find_original_file("final_model_comparison", "", ".csv", [])
    if final_file:
        dest_final = DEST_DIR / "final_model_comparison.csv"
        shutil.copy2(final_file, dest_final)
        try:
            df = pd.read_csv(dest_final)
            df = update_model_name_in_df(df, "Model")
            df.to_csv(dest_final, index=False)
            print("    [Updated] final_model_comparison.csv (Checked consistency)")
        except Exception as e:
            print(f"    [Error] Updating final comparison: {e}")

    print("\n=== All Tasks Completed ===")
    print(f"Files are gathered in: {DEST_DIR}")


if __name__ == "__main__":
    main()
