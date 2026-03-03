import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_DIR = DATA_DIR / "final_datasets_yearly"
OUTPUT_FILE = DATA_DIR / "final_data_top200.csv"
list_OUTPUT_FILE = DATA_DIR / "top200_stock_list.csv"

# フィルタ条件
TARGET_START_YEAR = 2020
TARGET_END_YEAR = 2025
MIN_DAYS_REQUIRED = 900
TOP_N = 200

# 異常値除外
PRICE_UPPER_LIMIT = 500000 

def select_top_stocks():
    print("--- Top 200 銘柄選定プロセス開始 (Encoding Fixed) ---")
    
    # 1. データ読み込み
    all_files = sorted(list(INPUT_DIR.glob("final_data_*.csv")))
    df_list = []
    print("データを読み込み中...")
    
    for f in tqdm(all_files):
        try:
            # codeを文字列として読み込む
            df = pd.read_csv(f, dtype={'code': str}, low_memory=False)
            df['Date'] = pd.to_datetime(df['Date'])
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    df_all = pd.concat(df_list, ignore_index=True)
    df_all['code'] = df_all['code'].astype(str).str.strip()
    
    initial_count = df_all['code'].nunique()
    print(f"元データ銘柄数: {initial_count}")

    # 2. 異常値クリーニング
    mask_valid_price = (df_all['Close'] > 0) & (df_all['Close'] < PRICE_UPPER_LIMIT)
    df_clean = df_all[mask_valid_price].copy()

    # 3. 期間フィルタ
    print(f"\n--- 品質フィルタ (2020-2025年 データ数 >= {MIN_DAYS_REQUIRED}) ---")
    mask_period = (df_clean['Date'].dt.year >= TARGET_START_YEAR) & \
                  (df_clean['Date'].dt.year <= TARGET_END_YEAR)
    df_period = df_clean[mask_period]
    
    stock_counts = df_period.groupby('code').size()
    valid_codes = stock_counts[stock_counts >= MIN_DAYS_REQUIRED].index.tolist()
    print(f"✅ 品質条件クリア銘柄数: {len(valid_codes)} / {initial_count}")

    # 4. 流動性フィルタ
    print(f"\n--- 流動性フィルタ (平均売買代金 Top {TOP_N}) ---")
    df_valid = df_period[df_period['code'].isin(valid_codes)].copy()
    df_valid['TradingValue'] = df_valid['Close'] * df_valid['Volume']
    avg_trading_value = df_valid.groupby('code')['TradingValue'].mean()
    
    top_codes = avg_trading_value.sort_values(ascending=False).head(TOP_N).index.tolist()
    
    # 上位銘柄確認
    if 'Name' in df_valid.columns:
        name_map = df_valid.drop_duplicates('code').set_index('code')['Name']
        print(f"Top 5 銘柄 (文字化け確認):")
        for c in top_codes[:5]:
            print(f" - {c}: {name_map.get(c, 'Unknown')}")

    # 5. データセット保存 (文字化け対策: utf-8-sig)
    print("\n--- データセット保存 ---")
    df_final = df_clean[df_clean['code'].isin(top_codes)].copy()
    df_final = df_final.sort_values(['code', 'Date'])
    
    # ★修正点: encoding='utf-8-sig' を指定
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"Top 200データセット保存完了: {OUTPUT_FILE}")
    
    pd.Series(top_codes, name='code').to_csv(list_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"銘柄リスト保存完了: {list_OUTPUT_FILE}")

if __name__ == "__main__":
    select_top_stocks()