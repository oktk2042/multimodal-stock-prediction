from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ==========================================
# 設定: 今回確定した6銘柄
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRED_DIR = PROJECT_ROOT / "3_reports" / "final_consolidated_v2"

TARGET_STOCKS = {
    "High": ["7267", "7203"],  # ホンダ, トヨタ
    "Medium": ["6305", "6988"],  # 日立建機, 日東電
    "Low": ["2801", "2432"],  # キッコマン, ディーエヌエ
}

MODELS = {
    "Proposed": "predictions_MultiModalGatedTransformer.csv",
    "PatchTST": "predictions_PatchTST.csv",
    "Vanilla": "predictions_VanillaTransformer.csv",
}

# グラフ設定
plt.rcParams["font.family"] = "MS Gothic"
sns.set(style="whitegrid", font="MS Gothic")


def analyze_fixed_6stocks():
    results = []

    for model_name, filename in MODELS.items():
        path = PRED_DIR / filename
        if not path.exists():
            continue

        df = pd.read_csv(path)
        # コード列の自動検出
        c_col = "Code" if "Code" in df.columns else "code"
        df["str_code"] = df[c_col].astype(str).str.extract(r"(\d{4})")[0]

        for category, codes in TARGET_STOCKS.items():
            for code in codes:
                stock_data = df[df["str_code"] == code]
                if not stock_data.empty:
                    # 方向正解率 (Accuracy) の計算
                    actual_dir = np.sign(stock_data["Actual"] - stock_data["Current"])
                    pred_dir = np.sign(stock_data["Pred_Return"])
                    acc = (actual_dir == pred_dir).mean()

                    results.append({"Model": model_name, "Category": category, "Code": code, "Accuracy": acc})

    res_df = pd.DataFrame(results)

    # 銘柄別の一覧表示
    pivot_df = res_df.pivot_table(index=["Category", "Code"], columns="Model", values="Accuracy")
    print("\n=== 確定6銘柄のモデル別方向正解率 (Accuracy) ===")
    print(pivot_df.reindex(["High", "Medium", "Low"], level=0))


if __name__ == "__main__":
    analyze_fixed_6stocks()
