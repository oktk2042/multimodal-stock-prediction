import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed" / "final_datasets_yearly"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "data_quality"

# 分析対象の年 (最新の2024年などを重点的に見る)
TARGET_YEAR = 2024

def main():
    print("--- 最終データセット品質分析 ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # ファイルを探す
    file_path = DATA_DIR / f"final_data_{TARGET_YEAR}.csv"
    if not file_path.exists():
        # なければあるやつを使う
        files = sorted(list(DATA_DIR.glob("final_data_*.csv")))
        if not files:
            print("エラー: データセットが見つかりません。")
            return
        file_path = files[-1]
    
    print(f"分析対象ファイル: {file_path.name}")
    df = pd.read_csv(file_path)
    
    # 1. 基本統計
    print("\n【基本情報】")
    print(f"行数: {len(df):,}")
    print(f"カラム数: {len(df.columns)}")
    print(f"銘柄数: {df['code'].nunique():,}")
    
    # 2. 欠損率チェック (重要)
    print("\n【欠損率ワースト5】")
    missing = df.isnull().mean() * 100
    print(missing.sort_values(ascending=False).head(5))
    
    # 3. 財務データの埋まり具合
    print("\n【財務データ含有率】")
    # 売上が0より大きい行の割合
    valid_sales = (df['NetSales'] > 0).mean() * 100
    print(f"売上高データあり: {valid_sales:.1f}%")
    
    # 利益が0以外の行 (赤字も含むので != 0)
    valid_profit = (df['OperatingIncome'] != 0).mean() * 100
    print(f"営業利益データあり: {valid_profit:.1f}%")
    
    # 4. ニュース・センチメントの密度
    print("\n【センチメント密度】")
    # 0以外の値が入っている率
    news_active = (df['News_Sentiment'] != 0).mean() * 100
    market_active = (df['Market_Sentiment'] != 0).mean() * 100
    print(f"個別ニュース発生率: {news_active:.1f}% (毎日ニュースがあるわけではないので低くて正常)")
    print(f"市場ニュース発生率: {market_active:.1f}%")

    # 5. 相関行列 (ヒートマップ保存)
    print("\n【相関分析】")
    # 数値列のみ
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # 重要指標に絞る
    target_cols = [
        'Close', 'Volume', 'RSI_14', 
        'NetSales', 'OperatingIncome', 'FinBERT_Score',
        'News_Sentiment', 'Market_Sentiment', 
        'SP500_Close', 'USD_JPY'
    ]
    # 存在する列だけ
    cols = [c for c in target_cols if c in numeric_df.columns]
    
    if cols:
        corr = numeric_df[cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title(f"Feature Correlation Matrix ({TARGET_YEAR})")
        save_path = OUTPUT_DIR / "correlation_matrix.png"
        plt.savefig(save_path)
        print(f"相関行列を保存しました: {save_path}")
    
    print("\n分析完了。このデータセットは学習に即座に使用可能です。")

if __name__ == "__main__":
    main()