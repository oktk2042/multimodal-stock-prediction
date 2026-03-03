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

# --- 入力ファイル群 ---
BASE_PRICE_FILE = DATA_DIR / "stock_data_features_v1.csv"
META_DATA_FILE = DATA_DIR / "integrated_financial_and_stock_data_v2.csv"
FINBERT_FILE = DATA_DIR / "edinet_features_finbert.csv"
LLM_FILE     = DATA_DIR / "edinet_features_llm.csv"
NEWS_FILE    = DATA_DIR / "news_sentiment_features.csv"

# --- 出力設定 ---
# 年別CSVを保存するフォルダ
OUTPUT_DIR_YEARLY = DATA_DIR / "final_datasets_yearly"
# Excelファイル (シート分け)
OUTPUT_FILE_XLSX = DATA_DIR / "final_model_input_dataset.xlsx"

# ==========================================
# 2. ユーティリティ関数
# ==========================================
def read_csv_safe(file_path):
    """エンコーディングを自動判別してCSVを読み込む"""
    encodings = ['utf-8', 'cp932', 'shift_jis', 'utf-16']
    for enc in encodings:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"ファイル {file_path} の読み込みに失敗しました。エンコーディングを確認してください。")

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("--- 最終データセット統合プロセス開始 ---")

    # 1. ベースデータの読み込み
    if not BASE_PRICE_FILE.exists():
        print(f"エラー: ベースデータが見つかりません {BASE_PRICE_FILE}")
        return
    
    print(f"ベースデータを読み込み中: {BASE_PRICE_FILE.name}")
    try:
        df_base = read_csv_safe(BASE_PRICE_FILE)
    except Exception as e:
        print(e)
        return
    
    # 日付とコードの統一
    if 'Date' in df_base.columns:
        df_base['Date'] = pd.to_datetime(df_base['Date'])
    
    if 'Code' in df_base.columns:
        df_base.rename(columns={'Code': 'code'}, inplace=True)
    
    print(f"  -> {len(df_base)} 行読み込み完了")

    # 2. メタデータの読み込み
    if META_DATA_FILE.exists():
        print(f"メタデータを読み込み中: {META_DATA_FILE.name}")
        df_meta = read_csv_safe(META_DATA_FILE)
        
        if 'periodEnd' in df_meta.columns:
            df_meta['periodEnd'] = pd.to_datetime(df_meta['periodEnd'])
        if 'submitDateTime' in df_meta.columns:
            df_meta['Date'] = pd.to_datetime(df_meta['submitDateTime']).dt.normalize()
        elif 'Date' in df_meta.columns:
            df_meta['Date'] = pd.to_datetime(df_meta['Date'])
    else:
        print("警告: メタデータが見つかりません。")
        df_meta = pd.DataFrame()

    # 3. 特徴量の読み込み
    print("特徴量ファイルを読み込み中...")
    df_finbert = read_csv_safe(FINBERT_FILE) if FINBERT_FILE.exists() else pd.DataFrame()
    df_llm = read_csv_safe(LLM_FILE) if LLM_FILE.exists() else pd.DataFrame()
    df_news = read_csv_safe(NEWS_FILE) if NEWS_FILE.exists() else pd.DataFrame()

    # --- 結合処理 ---
    
    # Step A: 決算特徴量 (FinBERT + LLM) を DocID でマージ
    if not df_finbert.empty and not df_llm.empty:
        df_reports = pd.merge(df_finbert, df_llm, on="DocID", how="outer")
    elif not df_finbert.empty:
        df_reports = df_finbert
    else:
        df_reports = df_llm

    if not df_reports.empty and not df_meta.empty:
        # Step B: DocID と (Code, PeriodEnd) の紐付け
        docid_to_period = {}
        if EDINET_DIR.exists():
            doc_folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
            for folder in tqdm(doc_folders, desc="Linking DocID"):
                csvs = list(folder.glob("**/*.csv"))
                if csvs:
                    match = re.search(r'_(\d{4}-\d{2}-\d{2})_', csvs[0].name)
                    if match:
                        docid_to_period[folder.name] = match.group(1)
        
        df_bridge = pd.DataFrame(list(docid_to_period.items()), columns=['DocID', 'periodEnd_str'])
        df_bridge['periodEnd'] = pd.to_datetime(df_bridge['periodEnd_str'])
        
        df_reports = pd.merge(df_reports, df_bridge, on='DocID', how='inner')
        df_fundamentals = pd.merge(df_meta, df_reports, on='periodEnd', how='left')
        
        cols_to_keep = ['code', 'Date', 'FinBERT_Score', 'NetSales_LLM', 'OperatingIncome_LLM', 'Sentiment_LLM', 'TextLength']
        cols_to_keep = [c for c in cols_to_keep if c in df_fundamentals.columns]
        
        df_fundamentals_clean = df_fundamentals[cols_to_keep].copy()
        df_fundamentals_clean = df_fundamentals_clean.groupby(['code', 'Date']).mean().reset_index()

        # Step C: ベースデータへのマージ
        print("ベースデータに決算情報をマージ中...")
        df_merged = pd.merge(df_base, df_fundamentals_clean, on=['code', 'Date'], how='left')
        
    else:
        df_merged = df_base.copy()

    # Step D: ニュース感情スコアのマージ
    if not df_news.empty:
        print("ニューススコアをマージ中...")
        if 'Code' in df_news.columns:
            df_news.rename(columns={'Code': 'code'}, inplace=True)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        
        df_news_agg = df_news.groupby(['code', 'Date'])['News_Sentiment'].mean().reset_index()
        df_merged = pd.merge(df_merged, df_news_agg, on=['code', 'Date'], how='left')

    # 4. 欠損値処理
    fill_cols = ['FinBERT_Score', 'Sentiment_LLM', 'News_Sentiment', 'NetSales_LLM', 'OperatingIncome_LLM']
    for col in fill_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].fillna(0.0)
    
    if 'TextLength' in df_merged.columns:
        df_merged['TextLength'] = df_merged['TextLength'].fillna(0)

    # 5. 年ごとに分割して保存
    print("\n--- 保存処理 ---")
    
    # 年情報を抽出
    df_merged['Year'] = df_merged['Date'].dt.year
    unique_years = sorted(df_merged['Year'].dropna().unique().astype(int))
    
    # 保存先フォルダ作成
    OUTPUT_DIR_YEARLY.mkdir(parents=True, exist_ok=True)
    
    # Excel Writer準備
    try:
        excel_writer = pd.ExcelWriter(OUTPUT_FILE_XLSX, engine='openpyxl')
        use_excel = True
    except Exception as e:
        print(f"Excel作成エラー(openpyxl不足など): {e}")
        use_excel = False

    for year in tqdm(unique_years, desc="Splitting by Year"):
        # その年のデータを抽出
        df_year = df_merged[df_merged['Year'] == year].copy()
        
        # 不要なYearカラムを削除
        df_year = df_year.drop(columns=['Year'])
        
        # 1. CSV保存 (年別ファイル)
        csv_filename = OUTPUT_DIR_YEARLY / f"final_data_{year}.csv"
        df_year.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # 2. Excel保存 (シート別)
        if use_excel:
            # シート名は文字列にする必要がある
            sheet_name = str(year)
            # 行数チェック (Excel制限回避)
            if len(df_year) < 1048000:
                df_year.to_excel(excel_writer, sheet_name=sheet_name, index=False)
            else:
                print(f"警告: {year}年のデータ({len(df_year)}行)はExcelの上限を超えるためシート保存をスキップしました。")

    if use_excel:
        print(f"Excelファイルを保存中: {OUTPUT_FILE_XLSX}")
        excel_writer.close()

    print("\n全工程完了！")
    print(f"年別CSVフォルダ: {OUTPUT_DIR_YEARLY}")
    print(f"Excelファイル: {OUTPUT_FILE_XLSX}")

if __name__ == "__main__":
    main()