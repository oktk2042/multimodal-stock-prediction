from pathlib import Path

import numpy as np
import pandas as pd

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"

# 読み込み
df = pd.read_csv(INPUT_FILE, low_memory=False)

# 1. 不要な 'Sec_nan' を削除
if "Sec_nan" in df.columns:
    df = df.drop(columns=["Sec_nan"])
    print("削除しました: Sec_nan")

# 2. 目的変数の作成 (Target: 翌日の対数収益率)
# AIに「何を予測させるか」の正解データを作っておきます
df.sort_values(["Code", "Date"], inplace=True)
df["Target_Return"] = df.groupby("Code")["Close"].transform(lambda x: np.log(x.shift(-1) / x))

# 欠損行（最終日など）を削除
df = df.dropna(subset=["Target_Return"])

# 保存
df.to_csv(INPUT_FILE, index=False, encoding="utf-8-sig")
print(f"最終データセット完成: {len(df)}行, {len(df.columns)}列")
print("追加された列: Target_Return")
