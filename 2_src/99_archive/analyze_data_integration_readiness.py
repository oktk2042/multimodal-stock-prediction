import pandas as pd
import numpy as np
import re
from pathlib import Path

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
EDINET_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

# 検証対象のファイル群
FILES = {
    "Stock": DATA_DIR / "stock_data_features_v1.csv",
    "Meta": DATA_DIR / "integrated_financial_and_stock_data_v2.csv",
    "FinBERT": DATA_DIR / "edinet_features_finbert.csv",
    "Financials": DATA_DIR / "edinet_features_financials_hybrid.csv",
    "News": DATA_DIR / "news_sentiment_historical.csv",
    "Sector": DATA_DIR / "stock_sector_info.csv",
    "Macro": DATA_DIR / "global_macro_features.csv"
}

def load_data(name, path, usecols=None):
    if not path.exists():
        print(f"❌ [Missing] {name}: {path.name} が見つかりません。")
        return None
    try:
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        print(f"✅ [Loaded] {name}: {len(df):,} 行")
        return df
    except Exception as e:
        print(f"❌ [Error] {name}: 読み込み失敗 ({e})")
        return None

def main():
    print("--- データ統合準備状況 分析レポート ---\n")

    # 1. データのロード
    df_stock = load_data("Stock", FILES["Stock"], usecols=['Date', 'Code'])
    df_meta = load_data("Meta", FILES["Meta"])
    df_finbert = load_data("FinBERT", FILES["FinBERT"], usecols=['DocID'])
    df_fin = load_data("Financials", FILES["Financials"], usecols=['DocID', 'NetSales', 'OperatingIncome'])
    df_news = load_data("News", FILES["News"], usecols=['Date', 'Code'])
    df_sector = load_data("Sector", FILES["Sector"], usecols=['code'])
    df_macro = load_data("Macro", FILES["Macro"], usecols=['Date'])

    if df_stock is None:
        print("\n⛔ ベースとなる株価データがないため、分析を中止します。")
        return

    # 2. キー項目の型統一
    print("\n--- キー項目の正規化 ---")
    
    # Stock
    if 'Code' in df_stock.columns:
        df_stock.rename(columns={'Code': 'code'}, inplace=True)
    df_stock['code'] = df_stock['code'].astype(str)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    stock_codes = set(df_stock['code'].unique())
    stock_dates = set(df_stock['Date'])
    print(f"Stock: {len(stock_codes)} 銘柄, {df_stock['Date'].min().date()} ～ {df_stock['Date'].max().date()}")

    # 3. [検証] 業種データ (Sector)
    print("\n--- [Check 1] 業種データとの結合 ---")
    if df_sector is not None:
        df_sector['code'] = df_sector['code'].astype(str)
        sector_codes = set(df_sector['code'].unique())
        
        missing_sector = stock_codes - sector_codes
        coverage = len(stock_codes - missing_sector) / len(stock_codes)
        print(f"業種データカバー率: {coverage:.1%}")
        if missing_sector:
            print(f"⚠️ 業種未定義の銘柄数: {len(missing_sector)} (例: {list(missing_sector)[:3]})")
        else:
            print("✅ 全銘柄の業種データが存在します。")

    # 4. [検証] マクロデータ (Macro)
    print("\n--- [Check 2] マクロデータとの結合 (Lag 1日考慮) ---")
    if df_macro is not None:
        df_macro['Date'] = pd.to_datetime(df_macro['Date'])
        # 結合時は1日ずらすため、マクロの日付を+1して検証
        macro_dates_shifted = set(df_macro['Date'] + pd.Timedelta(days=1))
        
        # 株価データの営業日に対して、マクロデータが存在するか
        # (マクロは土日もあるが株価はない。FFillするので直近があればOKだが、ここでは完全一致率を見る)
        valid_dates = stock_dates.intersection(macro_dates_shifted)
        date_coverage = len(valid_dates) / len(stock_dates)
        
        print(f"日付カバー率(完全一致): {date_coverage:.1%}")
        if date_coverage < 0.95:
            print("ℹ️ カバー率が100%でないのは休日ズレのため正常です。結合時のffillで補完されます。")
        print(f"マクロ期間: {df_macro['Date'].min().date()} ～ {df_macro['Date'].max().date()}")

    # 5. [検証] 財務データ (Financials & FinBERT -> Meta -> Stock)
    print("\n--- [Check 3] 財務データの結合パス (DocID -> Code/Date) ---")
    if df_fin is not None and df_meta is not None:
        # DocID の整合性
        fin_docids = set(df_fin['DocID'])
        if df_finbert is not None:
            bert_docids = set(df_finbert['DocID'])
            common_docs = fin_docids.intersection(bert_docids)
            print(f"FinBERTと数値データのDocID一致数: {len(common_docs):,} / {len(fin_docids):,} ({(len(common_docs)/len(fin_docids)):.1%})")
        
        # EDINETフォルダからDocIDとPeriodEndの対応表を作成 (V4のロジック再現)
        print("EDINETフォルダからDocIDマップを作成中(高速スキャン)...")
        docid_to_period = {}
        if EDINET_DIR.exists():
            folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
            for f in folders:
                # ファイル名シミュレーション
                # globは遅いので簡易的にフォルダ名がDocIDである前提と、中の日付文字列検索
                # ここでは正確性のためV4と同じくファイル名を見る
                try:
                    files = list(f.glob("XBRL_TO_CSV/*.csv"))
                    if not files:
                        files = list(f.glob("*.csv"))
                    if files:
                        m = re.search(r'_(\d{4}-\d{2}-\d{2})_', files[0].name)
                        if m:
                            docid_to_period[f.name] = m.group(1)
                except Exception:
                    continue
        
        print(f"  -> マッピング可能フォルダ数: {len(docid_to_period):,}")
        
        # 結合シミュレーション
        df_bridge = pd.DataFrame(list(docid_to_period.items()), columns=['DocID', 'periodEnd_str'])
        df_bridge['periodEnd'] = pd.to_datetime(df_bridge['periodEnd_str'])
        
        # Financials -> Bridge
        df_step1 = pd.merge(df_fin, df_bridge, on='DocID', how='inner')
        print(f"  -> 日付特定できた財務データ: {len(df_step1):,} 件")
        
        # Meta準備
        if 'periodEnd' in df_meta.columns:
            df_meta['periodEnd'] = pd.to_datetime(df_meta['periodEnd'])
        if 'Code' in df_meta.columns:
            df_meta.rename(columns={'Code': 'code'}, inplace=True)
        df_meta['code'] = df_meta['code'].astype(str)
        
        # Bridge -> Meta (PeriodEndで結合)
        df_meta_unique = df_meta[['code', 'periodEnd', 'submitDateTime']].drop_duplicates()
        if 'submitDateTime' in df_meta_unique.columns:
             df_meta_unique['Date'] = pd.to_datetime(df_meta_unique['submitDateTime']).dt.normalize()

        df_step2 = pd.merge(df_step1, df_meta_unique, on='periodEnd', how='inner')
        print(f"  -> 銘柄コード(Code)紐付け成功: {len(df_step2):,} 件")
        
        # Meta -> Stock (Code, Dateで結合)
        # 株価データの日付範囲に含まれるか
        df_step2 = df_step2[df_step2['code'].isin(stock_codes)]
        in_range = df_step2[ (df_step2['Date'] >= df_stock['Date'].min()) & (df_step2['Date'] <= df_stock['Date'].max()) ]
        
        print(f"  -> 株価データ期間内の財務データ: {len(in_range):,} 件")
        
        if len(in_range) > 0:
            print("✅ 財務データの結合パスは正常です。")
        else:
            print("❌ 警告: 財務データが株価データと一切結合できません。期間またはコードが不一致です。")

    # 6. [検証] ニュースデータ
    print("\n--- [Check 4] ニュースデータとの結合 ---")
    if df_news is not None:
        if 'Code' in df_news.columns:
            df_news.rename(columns={'Code': 'code'}, inplace=True)
        df_news['code'] = df_news['code'].astype(str)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        
        # マクロニュース
        macro_news = df_news[df_news['code'] == '9999']
        print(f"マクロニュース(9999): {len(macro_news):,} 件")
        
        # 個別ニュース
        indiv_news = df_news[df_news['code'] != '9999']
        valid_indiv = indiv_news[indiv_news['code'].isin(stock_codes)]
        print(f"対象銘柄のニュース: {len(valid_indiv):,} 件 (除外されたノイズ: {len(indiv_news) - len(valid_indiv)} 件)")

    print("\n" + "="*50)
    print("総合判定")
    print("="*50)
    
    issues = []
    if df_sector is None or (stock_codes - set(df_sector['code'].unique())):
        issues.append("業種データの欠損")
    if df_macro is None:
        issues.append("マクロデータの欠損")
    if len(in_range) < 1000:
        issues.append("財務データの紐付け失敗数が多い")
    
    if not issues:
        print("🎉 すべてのデータが正常に結合可能です！")
        print("   -> merge_features_final_v4.py を実行してください。")
    else:
        print("⚠️ 以下の懸念点があります:")
        for issue in issues:
            print(f"   - {issue}")
        print("   これらを許容できる場合のみ、結合に進んでください。")

if __name__ == "__main__":
    main()