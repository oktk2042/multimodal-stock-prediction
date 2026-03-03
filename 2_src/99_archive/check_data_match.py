import pandas as pd
import re
from pathlib import Path

# ==========================================
# 1. パス設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
EDINET_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

BASE_PRICE_FILE = DATA_DIR / "stock_data_features_v1.csv"
META_DATA_FILE = DATA_DIR / "integrated_financial_and_stock_data_v2.csv"
FINBERT_FILE = DATA_DIR / "edinet_features_finbert.csv"
NEWS_FILE    = DATA_DIR / "news_sentiment_features.csv"

def read_safe(path):
    """安全なCSV読み込み"""
    if not path.exists():
        print(f"[×] ファイルなし: {path.name}")
        return pd.DataFrame()
    for enc in ['utf-8', 'cp932', 'shift_jis']:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    return pd.DataFrame()

def main():
    print("=== データマッチング状況の調査開始 ===\n")

    # 1. 株価データ (ベース) の確認
    df_base = read_safe(BASE_PRICE_FILE)
    if df_base.empty: return
    
    # カラム名と型を統一
    if 'Code' in df_base.columns: df_base.rename(columns={'Code': 'code'}, inplace=True)
    if 'Date' in df_base.columns: df_base['Date'] = pd.to_datetime(df_base['Date'])
    df_base['code'] = df_base['code'].astype(str) # 文字列に統一

    print(f"【株価データ】")
    print(f"  - 行数: {len(df_base)}")
    print(f"  - 銘柄数: {df_base['code'].nunique()}")
    print(f"  - 期間: {df_base['Date'].min().date()} ～ {df_base['Date'].max().date()}")
    print(f"  - サンプルコード: {df_base['code'].unique()[:5]}")
    print("-" * 30)

    # 2. EDINET特徴量の確認 (DocID -> Code変換の成功率)
    print(f"【EDINET特徴量 (FinBERT)】")
    df_finbert = read_safe(FINBERT_FILE)
    df_meta = read_safe(META_DATA_FILE)

    if df_finbert.empty or df_meta.empty:
        print("  [×] EDINETまたはメタデータ不足")
    else:
        # メタデータの整備
        if 'periodEnd' in df_meta.columns: df_meta['periodEnd'] = pd.to_datetime(df_meta['periodEnd'])
        df_meta['code'] = df_meta['code'].astype(str)

        # フォルダからDocIDとPeriodEndの対応を取得
        docid_map = []
        if EDINET_DIR.exists():
            folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
            print(f"  - フォルダ数(DocID): {len(folders)}")
            
            for f in folders:
                csvs = list(f.glob("**/*.csv"))
                if csvs:
                    # ファイル名から日付抽出
                    m = re.search(r'_(\d{4}-\d{2}-\d{2})_', csvs[0].name)
                    if m:
                        docid_map.append({'DocID': f.name, 'periodEnd': pd.to_datetime(m.group(1))})
        
        df_bridge = pd.DataFrame(docid_map)
        
        # 結合シミュレーション
        # FinBERT -> Bridge(期間) -> Meta(コード)
        df_step1 = pd.merge(df_finbert, df_bridge, on='DocID', how='inner')
        df_step2 = pd.merge(df_step1, df_meta, on='periodEnd', how='inner') # Codeを取得

        print(f"  - FinBERTスコア件数: {len(df_finbert)}")
        print(f"  - 銘柄特定できた件数: {len(df_step2)}")
        if not df_step2.empty:
            print(f"  - 特定できた銘柄数: {df_step2['code'].nunique()}")
            
            # 株価データとのマッチングチェック
            stock_codes = set(df_base['code'])
            edinet_codes = set(df_step2['code'])
            common = stock_codes & edinet_codes
            print(f"  - ★株価データに含まれる銘柄数: {len(common)} / {len(edinet_codes)}")
            
            # 日付のマッチングチェック (直近結合)
            # 決算発表日(Date) と 株価日付(Date) が完全一致するか？
            if 'Date' in df_step2.columns: # Meta由来の提出日
                df_step2['Date'] = pd.to_datetime(df_step2['Date']).dt.normalize()
                
                merged_exact = pd.merge(df_base, df_step2, on=['code', 'Date'], how='inner')
                print(f"  - ★日付・銘柄が完全一致する行数: {len(merged_exact)} (これが0に近いと問題)")
    print("-" * 30)

    # 3. ニュースデータの確認
    print(f"【ニュースデータ】")
    df_news = read_safe(NEWS_FILE)
    if not df_news.empty:
        if 'Code' in df_news.columns: df_news.rename(columns={'Code': 'code'}, inplace=True)
        df_news['code'] = df_news['code'].astype(str)
        df_news['Date'] = pd.to_datetime(df_news['Date'])

        print(f"  - ニュース件数: {len(df_news)}")
        print(f"  - ニュースがある銘柄数: {df_news['code'].nunique()}")
        
        # 株価データとのマッチング
        stock_codes = set(df_base['code'])
        news_codes = set(df_news['code'])
        common_news = stock_codes & news_codes
        print(f"  - ★株価データに含まれる銘柄数: {len(common_news)}")

        merged_news = pd.merge(df_base, df_news, on=['code', 'Date'], how='inner')
        print(f"  - ★日付・銘柄が完全一致する行数: {len(merged_news)}")
    else:
        print("  [×] ニュースデータなし")

if __name__ == "__main__":
    main()