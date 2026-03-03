import pandas as pd
from pathlib import Path

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "1_data" / "processed" / "final_datasets_yearly"

def check_anomalies():
    print("データ異常チェックを開始します...")
    
    # 全ファイルを対象にする
    files = sorted(list(INPUT_DIR.glob("final_data_*.csv")))
    if not files:
        print("❌ エラー: データファイルが見つかりません。")
        return
    
    print(f"対象ファイル数: {len(files)}")
    
    found_high_price = False
    
    # 処理時間を短縮するため、最新のファイルから順にチェック
    for f in reversed(files):
        print(f"Checking {f.name} ...")
        # DtypeWarningが出ないようlow_memory=False
        df = pd.read_csv(f, low_memory=False)
        
        # 1. 株価異常値 (1000万円以上をチェック)
        # 日本株で1000万円を超える銘柄はほぼないので、これがあれば異常
        high_price_df = df[df['Close'] > 10000000] 
        if not high_price_df.empty:
            print(f"\n🚨 {f.name} に株価が1,000万円を超える行が {len(high_price_df)} 件あります！")
            print("--- 異常値サンプル ---")
            print(high_price_df[['Date', 'code', 'Name', 'Close', 'Volume']].head(5))
            found_high_price = True
            
        # 2. 売上高ゼロのサンプル確認
        zero_sales_df = df[df['NetSales'] == 0]
        if not zero_sales_df.empty and f == files[-1]: # 最新ファイルだけで確認
            print(f"\nℹ️ {f.name} に売上高が0の行が {len(zero_sales_df):,} 件あります。")
            print(f"   (全データの {len(zero_sales_df)/len(df)*100:.1f}%)")
            print(f"   対象銘柄例: {zero_sales_df['code'].unique()[:50]}")
            
        if found_high_price:
            print("\n異常値が見つかりました。確認してください。")
            break

    if not found_high_price:
        print("\n✅ 株価の異常値（1,000万円超）は見つかりませんでした。")

if __name__ == "__main__":
    check_anomalies()