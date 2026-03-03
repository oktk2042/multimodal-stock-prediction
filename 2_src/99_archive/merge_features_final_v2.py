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
NEWS_FILE    = DATA_DIR / "news_sentiment_features.csv"

# 出力設定
OUTPUT_DIR_YEARLY = DATA_DIR / "final_datasets_yearly"
OUTPUT_FILE_XLSX = DATA_DIR / "final_model_input_dataset.xlsx"

# ==========================================
# 2. ユーティリティ
# ==========================================
def read_csv_safe(file_path):
    """エンコーディング自動判別読み込み"""
    if not file_path.exists():
        return pd.DataFrame()
    for enc in ['utf-8', 'cp932', 'shift_jis', 'utf-16']:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except:
            continue
    return pd.DataFrame()

def main():
    print("--- 最終データセット統合 (Forward Fill版) 開始 ---")

    # 1. 株価データの読み込み
    print(f"株価データを読み込み中: {BASE_PRICE_FILE.name}")
    df_base = read_csv_safe(BASE_PRICE_FILE)
    if df_base.empty:
        print("エラー: 株価データが読み込めません")
        return

    # カラム名・型統一
    if 'Code' in df_base.columns: df_base.rename(columns={'Code': 'code'}, inplace=True)
    if 'Date' in df_base.columns: df_base['Date'] = pd.to_datetime(df_base['Date'])
    df_base['code'] = df_base['code'].astype(str)
    
    # ソートしておく（Forward Fillのために重要）
    df_base = df_base.sort_values(['code', 'Date'])
    print(f"  -> {len(df_base)} 行")

    # 2. 特徴量の準備 (EDINET)
    print("EDINET特徴量を準備中...")
    df_finbert = read_csv_safe(FINBERT_FILE)
    df_llm = read_csv_safe(LLM_FILE)
    df_meta = read_csv_safe(META_DATA_FILE)

    if not df_finbert.empty and not df_meta.empty:
        # メタデータの整備
        if 'periodEnd' in df_meta.columns: df_meta['periodEnd'] = pd.to_datetime(df_meta['periodEnd'])
        if 'submitDateTime' in df_meta.columns: 
            df_meta['Date'] = pd.to_datetime(df_meta['submitDateTime']).dt.normalize()
        df_meta['code'] = df_meta['code'].astype(str)

        # FinBERT + LLM 結合
        if not df_llm.empty:
            df_reports = pd.merge(df_finbert, df_llm, on="DocID", how="outer")
        else:
            df_reports = df_finbert

        # DocID -> PeriodEnd 辞書作成
        docid_to_period = {}
        if EDINET_DIR.exists():
            folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
            for f in tqdm(folders, desc="Linking DocID"):
                csvs = list(f.glob("**/*.csv"))
                if csvs:
                    m = re.search(r'_(\d{4}-\d{2}-\d{2})_', csvs[0].name)
                    if m: docid_to_period[f.name] = m.group(1)
        
        # マッピング適用
        df_bridge = pd.DataFrame(list(docid_to_period.items()), columns=['DocID', 'periodEnd_str'])
        df_bridge['periodEnd'] = pd.to_datetime(df_bridge['periodEnd_str'])
        
        # レポートに日付(PeriodEnd)を付与
        df_reports = pd.merge(df_reports, df_bridge, on='DocID', how='inner')
        
        # レポートにCodeと発表日(Date)を付与 (Meta経由)
        # ここで drop_duplicates をして重複爆発を防ぐ
        df_meta_unique = df_meta[['code', 'periodEnd', 'Date']].drop_duplicates()
        df_fundamentals = pd.merge(df_meta_unique, df_reports, on='periodEnd', how='inner') # innerにして確実にCodeがあるものだけ残す
        
        # 必要なカラムに絞る
        cols = ['code', 'Date', 'FinBERT_Score', 'NetSales_LLM', 'OperatingIncome_LLM', 'Sentiment_LLM']
        cols = [c for c in cols if c in df_fundamentals.columns]
        df_fund_clean = df_fundamentals[cols].copy()
        
        # 同日に複数ある場合は平均
        df_fund_clean = df_fund_clean.groupby(['code', 'Date']).mean().reset_index()
        
        print(f"  -> 結合可能な決算データ: {len(df_fund_clean)} 件")
        
        # ★ここが重要: ベースデータにマージ
        df_merged = pd.merge(df_base, df_fund_clean, on=['code', 'Date'], how='left')
        
        # ★Forward Fill (前方穴埋め) の実施
        # 銘柄ごとにグループ化し、時間を進めながら「直近の決算スコア」で埋める
        print("決算スコアを前方穴埋め(Forward Fill)中...")
        fill_cols = ['FinBERT_Score', 'Sentiment_LLM', 'NetSales_LLM', 'OperatingIncome_LLM']
        fill_cols = [c for c in fill_cols if c in df_merged.columns]
        
        # ffillを実行
        df_merged[fill_cols] = df_merged.groupby('code')[fill_cols].ffill()
        
        # 最初の決算より前は NaN になるので 0 で埋める
        df_merged[fill_cols] = df_merged[fill_cols].fillna(0)
        
    else:
        print("警告: EDINETデータ不足のためスキップ")
        df_merged = df_base.copy()

    # 3. ニュースデータのマージ
    print("ニュースデータをマージ中...")
    df_news = read_csv_safe(NEWS_FILE)
    if not df_news.empty:
        if 'Code' in df_news.columns: df_news.rename(columns={'Code': 'code'}, inplace=True)
        df_news['code'] = df_news['code'].astype(str)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        
        # 重複排除
        df_news_agg = df_news.groupby(['code', 'Date'])['News_Sentiment'].mean().reset_index()
        
        # マージ (ニュースはスポット的なので ffill しない)
        df_merged = pd.merge(df_merged, df_news_agg, on=['code', 'Date'], how='left')
        df_merged['News_Sentiment'] = df_merged['News_Sentiment'].fillna(0)

    # 4. 保存処理 (年別CSV & Excel)
    print("\n--- 保存処理 ---")
    OUTPUT_DIR_YEARLY.mkdir(parents=True, exist_ok=True)
    
    df_merged['Year'] = df_merged['Date'].dt.year
    unique_years = sorted(df_merged['Year'].dropna().unique().astype(int))

    # Excel Writer
    try:
        excel_writer = pd.ExcelWriter(OUTPUT_FILE_XLSX, engine='openpyxl')
        use_excel = True
    except:
        use_excel = False

    for year in tqdm(unique_years, desc="Saving"):
        df_year = df_merged[df_merged['Year'] == year].copy().drop(columns=['Year'])
        
        # CSV保存
        df_year.to_csv(OUTPUT_DIR_YEARLY / f"final_data_{year}.csv", index=False, encoding='utf-8-sig')
        
        # Excel保存
        if use_excel and len(df_year) < 1000000:
            df_year.to_excel(excel_writer, sheet_name=str(year), index=False)

    if use_excel:
        excel_writer.close()

    print("\n全工程完了！")
    print(f"CSVフォルダ: {OUTPUT_DIR_YEARLY}")
    
    # 結果確認
    sample = df_merged[df_merged['FinBERT_Score'] != 0].head()
    if not sample.empty:
        print("--- 穴埋め後のデータサンプル ---")
        print(sample[['Date', 'code', 'Close', 'FinBERT_Score']].head(10))

if __name__ == "__main__":
    main()