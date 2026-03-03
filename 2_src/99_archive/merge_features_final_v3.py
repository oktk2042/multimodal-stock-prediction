import pandas as pd
import numpy as np
import re
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. パス設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
EDINET_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

# 入力ファイル
BASE_PRICE_FILE = DATA_DIR / "stock_data_features_v1.csv"
META_DATA_FILE = DATA_DIR / "integrated_financial_and_stock_data_v2.csv"
FINBERT_FILE = DATA_DIR / "edinet_features_finbert.csv"
LLM_FILE     = DATA_DIR / "edinet_features_llm.csv"

# 巨大ニューススコアファイル
NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"

# 出力設定
OUTPUT_DIR_YEARLY = DATA_DIR / "final_datasets_yearly"
OUTPUT_FILE_XLSX = DATA_DIR / "final_model_input_dataset.xlsx"

# ==========================================
# 2. ユーティリティ
# ==========================================
def read_csv_safe(file_path):
    if not file_path.exists():
        return pd.DataFrame()
    # 巨大ファイル対策: low_memory=Falseで読み込み
    for enc in ['utf-8', 'utf-8-sig', 'cp932', 'shift_jis']:
        try:
            return pd.read_csv(file_path, encoding=enc, low_memory=False)
        except:
            continue
    return pd.DataFrame()

def main():
    print("--- 最終データセット統合 (V3: 完全版) 開始 ---")

    # 1. 株価データの読み込み
    print(f"株価データを読み込み中: {BASE_PRICE_FILE.name}")
    df_base = read_csv_safe(BASE_PRICE_FILE)
    if df_base.empty:
        print("エラー: 株価データが読み込めません")
        return

    if 'Code' in df_base.columns: df_base.rename(columns={'Code': 'code'}, inplace=True)
    if 'Date' in df_base.columns: df_base['Date'] = pd.to_datetime(df_base['Date'])
    df_base['code'] = df_base['code'].astype(str)
    df_base = df_base.sort_values(['code', 'Date'])
    
    print(f"  -> {len(df_base):,} 行")

    # 2. 決算データ (EDINET) の準備 & 結合
    print("決算スコアを準備中...")
    df_finbert = read_csv_safe(FINBERT_FILE)
    df_llm = read_csv_safe(LLM_FILE)
    df_meta = read_csv_safe(META_DATA_FILE)
    
    if not df_finbert.empty and not df_meta.empty:
        # メタデータ整備
        if 'periodEnd' in df_meta.columns: df_meta['periodEnd'] = pd.to_datetime(df_meta['periodEnd'])
        if 'submitDateTime' in df_meta.columns: 
            df_meta['Date'] = pd.to_datetime(df_meta['submitDateTime']).dt.normalize()
        df_meta['code'] = df_meta['code'].astype(str)
        
        # FinBERT + LLM 結合
        df_reports = pd.merge(df_finbert, df_llm, on="DocID", how="outer") if not df_llm.empty else df_finbert
        
        # DocID -> PeriodEnd マッピング (フォルダ名から)
        docid_to_period = {}
        if EDINET_DIR.exists():
            folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
            # 高速化のためtqdmなしで一気に処理
            for f in folders:
                csvs = list(f.glob("**/*.csv"))
                if csvs:
                    m = re.search(r'_(\d{4}-\d{2}-\d{2})_', csvs[0].name)
                    if m: docid_to_period[f.name] = m.group(1)
        
        df_bridge = pd.DataFrame(list(docid_to_period.items()), columns=['DocID', 'periodEnd_str'])
        df_bridge['periodEnd'] = pd.to_datetime(df_bridge['periodEnd_str'])
        
        # 結合: Report -> Bridge -> Meta
        df_reports = pd.merge(df_reports, df_bridge, on='DocID', how='inner')
        df_meta_unique = df_meta[['code', 'periodEnd', 'Date']].drop_duplicates()
        df_fundamentals = pd.merge(df_meta_unique, df_reports, on='periodEnd', how='inner')
        
        # 日次集計 (同日発表は平均)
        cols = ['code', 'Date', 'FinBERT_Score', 'NetSales_LLM', 'OperatingIncome_LLM', 'Sentiment_LLM']
        cols = [c for c in cols if c in df_fundamentals.columns]
        df_fund_clean = df_fundamentals[cols].groupby(['code', 'Date']).mean().reset_index()
        
        # ベースにマージ
        df_merged = pd.merge(df_base, df_fund_clean, on=['code', 'Date'], how='left')
        
        # Forward Fill (次の決算まで値を維持)
        print("  -> 決算情報をForward Fill中...")
        fill_cols = ['FinBERT_Score', 'Sentiment_LLM', 'NetSales_LLM', 'OperatingIncome_LLM']
        fill_cols = [c for c in fill_cols if c in df_merged.columns]
        df_merged[fill_cols] = df_merged.groupby('code')[fill_cols].ffill()
        df_merged[fill_cols] = df_merged[fill_cols].fillna(0) # 開始前は0
    else:
        print("  -> 決算データ不足のためスキップ")
        df_merged = df_base.copy()

    # 3. ニュースデータの処理
    print("ニュースデータを処理中 (マクロ/個別 分離)...")
    if NEWS_FILE.exists():
        df_news = read_csv_safe(NEWS_FILE)
        
        if 'Code' in df_news.columns: df_news.rename(columns={'Code': 'code'}, inplace=True)
        df_news['code'] = df_news['code'].astype(str)
        df_news['Date'] = pd.to_datetime(df_news['Date'], errors='coerce')
        df_news = df_news.dropna(subset=['Date'])
        
        # (A) マクロニュース (Code='9999')
        df_macro = df_news[df_news['code'] == '9999'].copy()
        if not df_macro.empty:
            print(f"  - マクロニュース: {len(df_macro):,} 件 -> 全銘柄のMarket_Sentimentへ")
            # 日次集計
            df_macro_daily = df_macro.groupby('Date')['News_Sentiment'].mean().reset_index()
            df_macro_daily.rename(columns={'News_Sentiment': 'Market_Sentiment'}, inplace=True)
            
            # 全データの 'Date' にマージ
            df_merged = pd.merge(df_merged, df_macro_daily, on='Date', how='left')
            df_merged['Market_Sentiment'] = df_merged['Market_Sentiment'].fillna(0)
        else:
            df_merged['Market_Sentiment'] = 0

        # (B) 個別銘柄ニュース (Code!='9999')
        df_individual = df_news[df_news['code'] != '9999'].copy()
        if not df_individual.empty:
            print(f"  - 個別ニュース: {len(df_individual):,} 件 -> 各銘柄のNews_Sentimentへ")
            # 日次・コード別集計
            df_indiv_daily = df_individual.groupby(['code', 'Date'])['News_Sentiment'].mean().reset_index()
            
            # マージ
            df_merged = pd.merge(df_merged, df_indiv_daily, on=['code', 'Date'], how='left')
            df_merged['News_Sentiment'] = df_merged['News_Sentiment'].fillna(0)
        else:
            df_merged['News_Sentiment'] = 0
            
    else:
        print("ニュースファイルが見つかりません")
        df_merged['Market_Sentiment'] = 0
        df_merged['News_Sentiment'] = 0

    # 4. 保存
    print("\n--- 保存処理 ---")
    OUTPUT_DIR_YEARLY.mkdir(parents=True, exist_ok=True)
    
    df_merged['Year'] = df_merged['Date'].dt.year
    unique_years = sorted(df_merged['Year'].dropna().unique().astype(int))

    # Excel Writer準備
    try:
        excel_writer = pd.ExcelWriter(OUTPUT_FILE_XLSX, engine='openpyxl')
        use_excel = True
    except:
        print("Excelライブラリ不足のためCSVのみ保存します")
        use_excel = False

    for year in tqdm(unique_years, desc="Saving Yearly Files"):
        df_year = df_merged[df_merged['Year'] == year].copy().drop(columns=['Year'])
        
        # CSV保存
        df_year.to_csv(OUTPUT_DIR_YEARLY / f"final_data_{year}.csv", index=False, encoding='utf-8-sig')
        
        # Excel保存 (行数制限チェック)
        if use_excel and len(df_year) < 1000000:
            df_year.to_excel(excel_writer, sheet_name=str(year), index=False)

    if use_excel:
        excel_writer.close()

    print("\n全工程完了！")
    print(f"CSV出力先: {OUTPUT_DIR_YEARLY}")
    print(f"Excel出力先: {OUTPUT_FILE_XLSX}")

if __name__ == "__main__":
    main()