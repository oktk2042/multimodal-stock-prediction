import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
# プロジェクトルートのパスを自動取得
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_DIR = DATA_DIR / "final_datasets_yearly"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "data_analysis"

# 出力ディレクトリ作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定（環境に合わせて変更してください。MacならJapan1, WindowsならMeiryoなど）
# sns.set(font='IPAGothic') 

def analyze_data_quality(df):
    """データ品質の統計的分析"""
    print("\n" + "="*50)
    print("1. データ基本統計量")
    print("="*50)
    print(f"総行数: {len(df):,}")
    print(f"総銘柄数: {df['code'].nunique()}")
    print(f"期間: {df['Date'].min()} 〜 {df['Date'].max()}")
    
    # 数値カラムの統計量
    desc = df.describe().T
    print("\n--- 数値データの要約統計量 (Top 10 rows) ---")
    print(desc[['mean', 'std', 'min', '50%', 'max']].head(10))
    
    # 欠損値の確認
    print("\n" + "="*50)
    print("2. 欠損値 (NaN) の確認")
    print("="*50)
    null_counts = df.isnull().sum()
    null_percent = (df.isnull().sum() / len(df)) * 100
    null_df = pd.concat([null_counts, null_percent], axis=1, keys=['Missing Count', 'Percent'])
    print(null_df[null_df['Missing Count'] > 0])
    
    if null_df['Missing Count'].sum() == 0:
        print("✅ 欠損値はありません (Amazing!)")
        
    # ゼロ値の割合（FinBERTや財務データなどのスパース性を確認）
    print("\n" + "="*50)
    print("3. スパース性確認 (値が0の割合)")
    print("="*50)
    # 0が含まれる可能性のある主要カラム
    check_cols = ['FinBERT_Score', 'News_Sentiment', 'NetSales', 'OperatingIncome']
    for col in check_cols:
        if col in df.columns:
            zero_count = (df[col] == 0).sum()
            zero_percent = (zero_count / len(df)) * 100
            print(f"{col}: 0の数 = {zero_count:,} ({zero_percent:.1f}%)")
            
    return desc

def analyze_time_continuity(df):
    """時系列の連続性チェック"""
    print("\n" + "="*50)
    print("4. 時系列連続性のチェック (Window処理用)")
    print("="*50)
    
    codes = df['code'].unique()
    sample_codes = codes[:5] # 全部は重いので最初の5銘柄でチェック
    
    print(f"サンプル {len(sample_codes)} 銘柄について日付の間隔を確認します...")
    
    for code in sample_codes:
        df_c = df[df['code'] == code].sort_values('Date')
        df_c['Date_Diff'] = df_c['Date'].diff().dt.days
        
        # 1日以上空いている箇所（土日祝は除く必要あるが、ここでは単純に大きなギャップを見る）
        gaps = df_c[df_c['Date_Diff'] > 5] # 5日以上空いてたら長期休暇かデータ欠損
        
        if len(gaps) > 0:
            print(f"⚠️ Code {code}: 5日以上のデータ抜けが {len(gaps)} 箇所あります。")
            print(gaps[['Date', 'Date_Diff']].head(3))
        else:
            print(f"Code {code}: 大きな欠損なし (OK)")

def plot_distributions(df):
    """分布の可視化"""
    print("\n" + "="*50)
    print("5. 分布の可視化 (画像保存)")
    print("="*50)
    
    # 1. FinBERT Scoreの分布
    if 'FinBERT_Score' in df.columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df['FinBERT_Score'], bins=50, kde=True)
        plt.title('Distribution of FinBERT Scores')
        plt.xlabel('Score')
        plt.savefig(OUTPUT_DIR / 'dist_finbert.png')
        print(f"保存完了: {OUTPUT_DIR / 'dist_finbert.png'}")
        plt.close()

    # 2. 売上高 (NetSales) の対数分布
    if 'NetSales' in df.columns:
        plt.figure(figsize=(10, 6))
        # 0や負の値を除外して対数をとる
        sales_log = np.log1p(df[df['NetSales'] > 0]['NetSales'])
        sns.histplot(sales_log, bins=50, kde=True)
        plt.title('Distribution of Log(NetSales)')
        plt.xlabel('Log(NetSales)')
        plt.savefig(OUTPUT_DIR / 'dist_netsales_log.png')
        print(f"保存完了: {OUTPUT_DIR / 'dist_netsales_log.png'}")
        plt.close()
        
    # 3. 相関行列 (主要な数値列のみ)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # 計算負荷軽減のため、主要なカラムに絞るか、サンプリングする
    target_cols = [c for c in numeric_cols if 'Sec_' not in c and 'is_' not in c] # セクターダミーなどは除く
    if len(target_cols) > 20:
        target_cols = target_cols[:20] # 多すぎると見えないので
        
    plt.figure(figsize=(12, 10))
    corr = df[target_cols].sample(frac=0.1, random_state=42).corr() # 10%サンプリングで計算
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix (Sampled)')
    plt.savefig(OUTPUT_DIR / 'correlation_matrix.png')
    print(f"保存完了: {OUTPUT_DIR / 'correlation_matrix.png'}")
    plt.close()

def main():
    print("データの読み込み中...")
    all_files = sorted(list(INPUT_DIR.glob("final_data_*.csv")))
    
    if not all_files:
        print("エラー: データファイルが見つかりません。")
        return

    df_list = []
    for f in tqdm(all_files):
        df_tmp = pd.read_csv(f)
        df_tmp['Date'] = pd.to_datetime(df_tmp['Date'])
        df_list.append(df_tmp)
    
    df_all = pd.concat(df_list, ignore_index=True)
    df_all = df_all.sort_values(['code', 'Date'])
    
    # 分析実行
    analyze_data_quality(df_all)
    analyze_time_continuity(df_all)
    plot_distributions(df_all)
    
    print("\n分析完了。 '3_reports/data_analysis/' を確認してください。")

if __name__ == "__main__":
    main()