import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# データ読み込み
PROJECT_ROOT = Path(".").resolve() # 適宜調整してください
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
SECTOR_FILE = DATA_DIR / "stock_sector_info.csv"

def plot_sector_distribution():
    if not SECTOR_FILE.exists():
        print("Sector file not found.")
        return

    df = pd.read_csv(SECTOR_FILE)
    
    # セクターごとのカウント
    sector_counts = df['Sector'].value_counts()
    
    # 可視化設定
    plt.figure(figsize=(12, 8))
    plt.rcParams['font.size'] = 12
    
    # カラーパレット
    colors = sns.color_palette('pastel')[0:len(sector_counts)]
    
    # 円グラフ
    wedges, texts, autotexts = plt.pie(
        sector_counts, 
        labels=sector_counts.index, 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=colors,
        pctdistance=0.85,
        wedgeprops=dict(width=0.5) # ドーナツ型にする
    )
    
    plt.setp(texts, size=10)
    plt.setp(autotexts, size=9, weight="bold")
    
    plt.title("Sector Distribution of Target 200 Stocks", fontsize=16)
    plt.tight_layout()
    
    # 保存
    save_path = PROJECT_ROOT / "3_reports" / "phase3_production" / "dataset_sector_distribution.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")

if __name__ == "__main__":
    plot_sector_distribution()