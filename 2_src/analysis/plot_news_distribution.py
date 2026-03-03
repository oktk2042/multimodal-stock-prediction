from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# === 日本語フォント設定 (Windows用) ===
matplotlib.rcParams["font.family"] = "MS Gothic"
sns.set(font="MS Gothic", style="whitegrid")

# データ読み込み (パスは適宜変更してください)
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
df = pd.read_csv(DATA_DIR / "analysis_results/news_stats_by_code.csv")

# プロット作成
plt.figure(figsize=(10, 6))

# ヒストグラムを描画
# 月間平均記事数の分布を見るのが適切です
sns.histplot(df["Articles_Per_Month"], bins=30, kde=True, color="skyblue", edgecolor="black")

plt.title("銘柄ごとの月間平均ニュース記事数分布", fontsize=14)
plt.xlabel("月間平均記事数 (Articles / Month)", fontsize=12)
plt.ylabel("銘柄数 (Frequency)", fontsize=12)

# 平均値と中央値の線
mean_val = df["Articles_Per_Month"].mean()
median_val = df["Articles_Per_Month"].median()

plt.axvline(mean_val, color="red", linestyle="--", label=f"平均値: {mean_val:.1f}")
plt.axvline(median_val, color="green", linestyle="-", label=f"中央値: {median_val:.1f}")

plt.legend()
plt.tight_layout()

# 保存
plt.savefig(DATA_DIR / "news_frequency_distribution.png", dpi=300)
plt.show()
