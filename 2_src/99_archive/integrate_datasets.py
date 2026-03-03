import pandas as pd
import re
import numpy as np
from pathlib import Path

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
FINANCIAL_RAW_FILE = DATA_DIR / "extracted_financial_data.csv"

# 出力ファイル
OUTPUT_FILE = DATA_DIR / "final_modeling_dataset.csv"

# ==========================================
# 2. 関数定義
# ==========================================

def get_submission_date(filename):
    """ファイル名から提出日を抽出"""
    dates = re.findall(r'(\d{4}-\d{2}-\d{2})', str(filename))
    if len(dates) >= 2:
        return dates[-1]
    return None

def process_financials(df_fin):
    """財務データを整形"""
    print("Processing financials...")
    
    def clean_val(x):
        try:
            return float(str(x).replace(',', '').replace('△', '-').replace('▲', '-'))
        except Exception:
            return np.nan
            
    df_fin['Value'] = df_fin['Value'].apply(clean_val)
    
    df_fin['SubmissionDate'] = df_fin['File'].apply(get_submission_date)
    df_fin['SubmissionDate'] = pd.to_datetime(df_fin['SubmissionDate'])
    df_fin = df_fin.dropna(subset=['SubmissionDate'])
    
    mapping = {
        'NetSales': ['NetSales', 'Revenue', '売上'],
        'OperatingIncome': ['Operating', '営業利益'],
        'NetIncome': ['NetIncome', 'ProfitLossAttributable', '当期純利益', '当期利益'],
        'NetAssets': ['NetAssets', 'Equity', '純資産'],
        'TotalAssets': ['TotalAssets', 'Assets', '総資産']
    }
    
    def map_item(row):
        eid = str(row['ElementID'])
        name = str(row['ItemName'])
        
        if 'NetSales' in eid or 'Revenue' in eid:
            return 'NetSales'
        if 'Operating' in eid:
            return 'OperatingIncome'
        if 'ProfitLossAttributable' in eid:
            return 'NetIncome'
        if 'NetAssets' in eid or 'EquityIFRS' in eid:
            return 'NetAssets'
        if 'TotalAssets' in eid or 'AssetsIFRS' in eid:
            return 'TotalAssets'
        
        for key, keywords in mapping.items():
            for kw in keywords:
                if kw in name:
                    return key
        return None

    df_fin['StandardItem'] = df_fin.apply(map_item, axis=1)
    df_fin = df_fin.dropna(subset=['StandardItem'])
    
    df_wide = df_fin.pivot_table(
        index=['Code', 'SubmissionDate'],
        columns='StandardItem',
        values='Value',
        aggfunc='max'
    ).reset_index()
    
    return df_wide.sort_values(['Code', 'SubmissionDate'])

def main():
    print("--- Loading Data ---")
    
    # 1. 株価データ (encoding指定なしで自動判定に任せるか、utf-8等試行)
    # 文字化けの原因が読み込み時の可能性もあるため、一般的ないくつかのエンコードを試す
    try:
        df_price = pd.read_csv(PRICE_FILE, encoding='utf-8')
    except Exception:
        try:
            df_price = pd.read_csv(PRICE_FILE, encoding='cp932')
        except Exception:
            df_price = pd.read_csv(PRICE_FILE, encoding='shift_jis')

    if 'code' in df_price.columns:
        print("Renaming 'code' column to 'Code'...")
        df_price = df_price.rename(columns={'code': 'Code'})
        
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    df_price['Code'] = df_price['Code'].astype(str)
    print(f"Price data: {len(df_price)} rows")

    # 2. ニュースデータ
    df_news_agg = None
    if NEWS_FILE.exists():
        df_news = pd.read_csv(NEWS_FILE, low_memory=False)
        if 'code' in df_news.columns:
            df_news = df_news.rename(columns={'code': 'Code'})
            
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Code'] = df_news['Code'].astype(str)
        
        df_news_agg = df_news.groupby(['Code', 'Date']).agg({
            'News_Sentiment': 'mean',
            'Title': 'count'
        }).reset_index().rename(columns={'Title': 'News_Count'})
        print(f"News data (aggregated): {len(df_news_agg)} rows")

    # 3. 財務データ
    if FINANCIAL_RAW_FILE.exists():
        df_fin_raw = pd.read_csv(FINANCIAL_RAW_FILE, dtype={'Code': str}, low_memory=False)
        df_fin_wide = process_financials(df_fin_raw)
        print(f"Financial data (wide): {len(df_fin_wide)} rows (events)")
    else:
        print("Error: Financial file missing.")
        return

    print("\n--- Merging Datasets ---")
    
    df_final = df_price.sort_values(['Code', 'Date'])

    # 上書き準備：既存のニュース・財務カラムを削除
    cols_to_remove = ['News_Sentiment', 'News_Count']
    cols_to_remove += [c for c in df_fin_wide.columns if c not in ['Code', 'SubmissionDate']]
    cols_to_remove += [f"Fin_{c}" for c in df_fin_wide.columns if c not in ['Code', 'SubmissionDate']]
    
    # 実際に存在するカラムだけ削除
    existing_cols = [c for c in cols_to_remove if c in df_final.columns]
    if existing_cols:
        print(f"Overwriting columns: {existing_cols}")
        df_final = df_final.drop(columns=existing_cols)

    # (A) ニュース統合
    if df_news_agg is not None:
        df_final = pd.merge(df_final, df_news_agg, on=['Code', 'Date'], how='left')
        df_final['News_Sentiment'] = df_final['News_Sentiment'].fillna(0)
        df_final['News_Count'] = df_final['News_Count'].fillna(0)

    # (B) 財務データ統合
    df_final = df_final.sort_values('Date')
    df_fin_wide = df_fin_wide.sort_values('SubmissionDate')
    
    df_final = pd.merge_asof(
        df_final.sort_values('Date'),
        df_fin_wide.sort_values('SubmissionDate'),
        left_on='Date',
        right_on='SubmissionDate',
        by='Code',
        direction='backward'
    )
    
    # 5. 指標計算
    print("\n--- Calculating Metrics ---")
    if 'MarketCap' in df_final.columns:
        if 'NetAssets' in df_final.columns:
            df_final['PBR'] = df_final.apply(lambda x: x['MarketCap'] / x['NetAssets'] if pd.notnull(x['NetAssets']) and x['NetAssets'] > 0 else np.nan, axis=1)
        if 'NetIncome' in df_final.columns:
            df_final['PER'] = df_final.apply(lambda x: x['MarketCap'] / x['NetIncome'] if pd.notnull(x['NetIncome']) and x['NetIncome'] > 0 else np.nan, axis=1)
        if 'NetSales' in df_final.columns:
             df_final['PSR'] = df_final.apply(lambda x: x['MarketCap'] / x['NetSales'] if pd.notnull(x['NetSales']) and x['NetSales'] > 0 else np.nan, axis=1)

    # カラム名整理
    new_fin_cols = [c for c in df_fin_wide.columns if c not in ['Code', 'SubmissionDate']]
    rename_map = {c: f"Fin_{c}" for c in new_fin_cols}
    df_final = df_final.rename(columns=rename_map)

    # 保存 (utf-8-sig で文字化け防止)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {OUTPUT_FILE}")
    print("Columns sample:", df_final.columns.tolist()[:10])

if __name__ == "__main__":
    main()