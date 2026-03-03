import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. パス設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
ID_MAP_FILE_PATH = DATA_DIR / "EDINET_Summary_v3.csv"

# 入力ファイル群
BASE_PRICE_FILE = DATA_DIR / "stock_data_features_v1.csv"
FINBERT_FILE    = DATA_DIR / "edinet_features_finbert.csv"
FINANCIALS_FILE = DATA_DIR / "edinet_features_financials_hybrid.csv" 
NEWS_FILE       = DATA_DIR / "news_sentiment_historical.csv"
SECTOR_FILE     = DATA_DIR / "stock_sector_info.csv"
MACRO_FILE      = DATA_DIR / "global_macro_features.csv"

# 出力設定
OUTPUT_DIR_YEARLY = DATA_DIR / "final_datasets_yearly"
OUTPUT_FILE_XLSX  = DATA_DIR / "final_model_input_dataset.xlsx"

# ==========================================
# 2. ユーティリティ
# ==========================================
def read_csv_safe(file_path):
    if not file_path.exists():
        return pd.DataFrame()
    for enc in ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis']:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False)
        except Exception:
            continue
    return pd.DataFrame()

def main():
    print("--- 最終データセット統合 (V5: DocID直接紐付け版) 開始 ---")

    # 1. 紐付けマップの読み込み
    print(f"1. 紐付けマップ({ID_MAP_FILE_PATH.name})を読み込み中...")
    if not ID_MAP_FILE_PATH.exists():
        print(f"❌ エラー: {ID_MAP_FILE_PATH} が見つかりません。")
        return
        
    df_map = read_csv_safe(ID_MAP_FILE_PATH)
    
    # 前処理: カラム名統一とコード変換
    if 'docID' in df_map.columns:
        df_map.rename(columns={'docID': 'DocID'}, inplace=True)
    if 'submitDateTime' in df_map.columns:
        df_map.rename(columns={'submitDateTime': 'Date'}, inplace=True)
    
    # secCode '75390' -> '7539' (strにして先頭4桁)
    df_map['secCode'] = df_map['secCode'].fillna('').astype(str).str.replace('.0', '', regex=False)
    df_map['code'] = df_map['secCode'].str[:4]
    
    # 必要な列だけ抽出 & 日付型変換
    df_map = df_map[['DocID', 'code', 'Date']].drop_duplicates(subset=['DocID'])
    df_map['Date'] = pd.to_datetime(df_map['Date']).dt.normalize()
    
    print(f"   -> 紐付け可能データ: {len(df_map):,} 件")

    # 2. 財務データ (Hybrid + FinBERT) の結合
    print("2. 財務データを準備中...")
    df_finbert = read_csv_safe(FINBERT_FILE)
    df_nums    = read_csv_safe(FINANCIALS_FILE)
    
    # DocIDで結合 (Hybrid数値をベースにFinBERTスコアを付与)
    df_financials = pd.merge(df_finbert, df_nums, on="DocID", how="outer")
    
    # さらにMapと結合して「銘柄コード」と「日付」を付与
    df_financials = pd.merge(df_financials, df_map, on="DocID", how="inner")
    
    # 必要な列: code, Date, 各種スコア
    fin_cols = ['code', 'Date', 'FinBERT_Score', 'NetSales', 'OperatingIncome']
    # 存在しない列はスキップ
    fin_cols = [c for c in fin_cols if c in df_financials.columns]
    df_financials = df_financials[fin_cols]
    
    # 同日に複数発表がある場合（訂正報告書など）は平均をとる
    df_financials = df_financials.groupby(['code', 'Date']).mean().reset_index()
    print(f"   -> 財務データ(銘柄・日付特定済): {len(df_financials):,} 件")

    # 3. 株価データの読み込み (ベース)
    print("3. 株価データを読み込み中...")
    df_base = read_csv_safe(BASE_PRICE_FILE)
    if df_base.empty:
        print("❌ エラー: 株価データがありません")
        return

    if 'Code' in df_base.columns:
        df_base.rename(columns={'Code': 'code'}, inplace=True)
    if 'Date' in df_base.columns:
        df_base['Date'] = pd.to_datetime(df_base['Date'])
    df_base['code'] = df_base['code'].astype(str)
    df_base = df_base.sort_values(['code', 'Date'])
    print(f"   -> 株価データ: {len(df_base):,} 行")

    # 4. ベースに財務データをマージ
    print("4. 株価データに財務データを結合中...")
    df_merged = pd.merge(df_base, df_financials, on=['code', 'Date'], how='left')
    
    # Forward Fill (次の決算まで値を維持)
    print("   -> Forward Fill 処理中...")
    fill_cols = ['FinBERT_Score', 'NetSales', 'OperatingIncome']
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged.groupby('code')[col].ffill()
            df_merged[col] = df_merged[col].fillna(0)

    # 5. 業種データの結合
    print("5. 業種データを結合中...")
    df_sector = read_csv_safe(SECTOR_FILE)
    if not df_sector.empty:
        df_sector['code'] = df_sector['code'].astype(str)
        # 必要なカラムだけ
        use_cols = ['code', 'Sector', 'MarketCap']
        df_sector = df_sector[[c for c in use_cols if c in df_sector.columns]]
        
        df_merged = pd.merge(df_merged, df_sector, on='code', how='left')
        
        # One-Hot Encoding
        if 'Sector' in df_merged.columns:
            df_merged['Sector'] = df_merged['Sector'].fillna('Unknown')
            df_merged = pd.get_dummies(df_merged, columns=['Sector'], prefix='Sec')
    else:
        print("⚠️ 業種データなし")

    # 6. グローバルマクロ (Lag 1)
    print("6. グローバルマクロを結合中...")
    df_macro = read_csv_safe(MACRO_FILE)
    if not df_macro.empty:
        df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        # 1日ずらす
        df_macro['Date'] = df_macro['Date'] + pd.Timedelta(days=1)
        
        df_merged = pd.merge(df_merged, df_macro, on='Date', how='left')
        
        # 欠損埋め
        macro_cols = [c for c in df_macro.columns if c != 'Date']
        df_merged[macro_cols] = df_merged[macro_cols].ffill()
    else:
        print("⚠️ マクロデータなし")

    # 7. ニュースデータ
    print("7. ニュースデータを結合中...")
    if NEWS_FILE.exists():
        df_news = read_csv_safe(NEWS_FILE)
        if 'Code' in df_news.columns:
            df_news.rename(columns={'Code': 'code'}, inplace=True)
        df_news['code'] = df_news['code'].astype(str)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        
        # マクロニュース
        df_macro_n = df_news[df_news['code'] == '9999'].copy()
        if not df_macro_n.empty:
            df_m_daily = df_macro_n.groupby('Date')['News_Sentiment'].mean().reset_index()
            df_m_daily.rename(columns={'News_Sentiment': 'Market_Sentiment'}, inplace=True)
            df_merged = pd.merge(df_merged, df_m_daily, on='Date', how='left')
            df_merged['Market_Sentiment'] = df_merged['Market_Sentiment'].fillna(0)
        else:
            df_merged['Market_Sentiment'] = 0
            
        # 個別ニュース
        df_indiv_n = df_news[df_news['code'] != '9999'].copy()
        if not df_indiv_n.empty:
            df_i_daily = df_indiv_n.groupby(['code', 'Date'])['News_Sentiment'].mean().reset_index()
            df_merged = pd.merge(df_merged, df_i_daily, on=['code', 'Date'], how='left')
            df_merged['News_Sentiment'] = df_merged['News_Sentiment'].fillna(0)
        else:
            df_merged['News_Sentiment'] = 0
    else:
        print("⚠️ ニュースデータなし")
        df_merged['Market_Sentiment'] = 0
        df_merged['News_Sentiment'] = 0

    # 8. 保存
    print("\n--- 保存処理 ---")
    OUTPUT_DIR_YEARLY.mkdir(parents=True, exist_ok=True)
    df_merged['Year'] = df_merged['Date'].dt.year
    unique_years = sorted(df_merged['Year'].dropna().unique().astype(int))
    
    # Excelライター
    use_excel = False
    try:
        excel_writer = pd.ExcelWriter(OUTPUT_FILE_XLSX, engine='openpyxl')
        use_excel = True
    except Exception:
        pass

    for year in tqdm(unique_years, desc="Saving"):
        df_year = df_merged[df_merged['Year'] == year].copy().drop(columns=['Year'])
        
        # CSV
        df_year.to_csv(OUTPUT_DIR_YEARLY / f"final_data_{year}.csv", index=False, encoding='utf-8-sig')
        
        # Excel
        if use_excel and len(df_year) < 1000000:
            df_year.to_excel(excel_writer, sheet_name=str(year), index=False)

    if use_excel:
        excel_writer.close()

    print("\n✅ 全工程完了！おめでとうございます。")
    print(f"最終データ形状: {df_merged.shape}")
    print(f"出力先: {OUTPUT_DIR_YEARLY}")

if __name__ == "__main__":
    main()