import pandas as pd
import numpy as np
from pathlib import Path

# ==========================================
# 1. パス設定 (ユーザー環境に完全準拠)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル群
FILES = {
    "Stock": DATA_DIR / "stock_data_features_v1.csv",
    "FinBERT": DATA_DIR / "edinet_features_finbert.csv",
    "Financials": DATA_DIR / "edinet_features_financials_hybrid.csv",
    "News": DATA_DIR / "news_sentiment_historical.csv",
    "Sector": DATA_DIR / "stock_sector_info.csv",
    "Macro": DATA_DIR / "global_macro_features.csv",
    "ID_Map": DATA_DIR / "EDINET_Summary_v3.csv"
}

def load_data(name, path, usecols=None):
    if not path.exists():
        print(f"❌ [Missing] {name}: {path.name} が見つかりません。")
        return None
    try:
        # ID_Mapなどの巨大ファイルは必要な列だけ読む
        df = pd.read_csv(path, usecols=usecols, low_memory=False)
        print(f"✅ [Loaded] {name}: {len(df):,} 行")
        return df
    except Exception as e:
        print(f"❌ [Error] {name}: 読み込み失敗 ({e})")
        return None

def main():
    print("--- 最終結合(V5) 準備状況 & 整合性分析 (完全版) ---\n")

    # 1. データのロード
    df_stock = load_data("Stock", FILES["Stock"], usecols=['Date', 'Code'])
    df_map = load_data("ID_Map", FILES["ID_Map"], usecols=['docID', 'secCode', 'submitDateTime'])
    df_fin = load_data("Financials", FILES["Financials"], usecols=['DocID', 'NetSales', 'OperatingIncome'])
    df_finbert = load_data("FinBERT", FILES["FinBERT"], usecols=['DocID', 'FinBERT_Score'])
    
    # 他のファイルの存在確認
    for name in ["News", "Sector", "Macro"]:
        if not FILES[name].exists():
            print(f"❌ [Missing] {name}: {FILES[name]}")
    
    if df_stock is None or df_map is None or df_fin is None:
        print("\n⛔ 必須ファイル(Stock, ID_Map, Financials)が不足しているため中止します。")
        return

    # 2. [検証] IDマップの有効性とコード変換
    print("\n--- [Check 1] IDマップによる紐付けシミュレーション ---")
    
    # Stock側のコード一覧
    if 'Code' in df_stock.columns:
        df_stock.rename(columns={'Code': 'code'}, inplace=True)
    df_stock['code'] = df_stock['code'].astype(str)
    stock_codes = set(df_stock['code'].unique())
    print(f"株価データの銘柄数: {len(stock_codes):,} (例: {list(stock_codes)[:3]})")
    
    # IDマップの前処理 & 変換ロジック検証
    # カラム名統一
    if 'docID' in df_map.columns:
        df_map.rename(columns={'docID': 'DocID'}, inplace=True)
    if 'submitDateTime' in df_map.columns:
        df_map.rename(columns={'submitDateTime': 'Date'}, inplace=True)
    
    # secCodeの欠損チェック
    missing_sec = df_map['secCode'].isnull().sum()
    if missing_sec > 0:
        print(f"ℹ️ secCodeが空の行: {missing_sec:,} 件 (これらは銘柄紐付けできません)")
    
    # 変換ロジック: 末尾1桁を削る（5桁->4桁）
    # ただし、floatで読まれている可能性も考慮して一度strにする
    df_map['secCode'] = df_map['secCode'].fillna('').astype(str).str.replace('.0', '', regex=False)
    
    # '75390' -> '7539' (先頭4桁を取得)
    df_map['code_processed'] = df_map['secCode'].str[:4]
    
    # 変換サンプルの表示
    sample_conversion = df_map[['secCode', 'code_processed']].drop_duplicates().head(5)
    print("コード変換サンプル:")
    print(sample_conversion.to_string(index=False))

    # マッチング率確認
    map_valid_codes = set(df_map['code_processed'].unique())
    overlap = stock_codes.intersection(map_valid_codes)
    coverage = len(overlap) / len(stock_codes)
    
    print(f"IDマップに含まれる対象銘柄数: {len(overlap):,} / {len(stock_codes):,} (カバー率: {coverage:.1%})")
    
    if coverage < 0.9:
        print("⚠️ 警告: IDマップでカバーできない銘柄が10%以上あります。上場廃止やコード変更の影響の可能性があります。")
    else:
        print("✅ IDマップのコード変換は正常に機能します。")

    # 3. [検証] 財務データの結合テスト (V5ロジック)
    print("\n--- [Check 2] 財務データ(DocID) -> 株価データ(Code) の到達率 ---")
    
    # DocIDの重複チェック (IDマップ側)
    dup_docs = df_map['DocID'].duplicated().sum()
    if dup_docs > 0:
        print(f"ℹ️ IDマップ内のDocID重複: {dup_docs:,} 件 -> 先頭を優先して重複削除します")
        df_map = df_map.drop_duplicates(subset=['DocID'])
    
    # 結合シミュレーション: Financials(Hybrid) + ID_Map
    df_fin_merged = pd.merge(df_fin, df_map[['DocID', 'code_processed', 'Date']], on='DocID', how='inner')
    print(f"IDマップと結合できた財務データ(数値): {len(df_fin_merged):,} / {len(df_fin):,} 件")
    
    # 日付型の変換と期間チェック
    df_fin_merged['Date'] = pd.to_datetime(df_fin_merged['Date']).dt.normalize()
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    
    min_date = df_stock['Date'].min()
    max_date = df_stock['Date'].max()
    
    # 株価データの期間・銘柄に含まれる有効データ
    in_range = df_fin_merged[
        (df_fin_merged['Date'] >= min_date) & 
        (df_fin_merged['Date'] <= max_date) &
        (df_fin_merged['code_processed'].isin(stock_codes))
    ]
    
    print(f"株価データ期間・銘柄に合致する有効財務データ数: {len(in_range):,} 件")
    
    # 4. 総合判定
    print("\n" + "="*50)
    print("総合判定 (V5 Readiness)")
    print("="*50)
    
    issues = []
    if coverage <= 0.8:
        issues.append(f"銘柄カバー率が低め ({coverage:.1%})")
    if len(in_range) <= 1000:
        issues.append("有効な財務データが少なすぎる")
    
    if not issues:
        print("🎉 合格: すべての条件をクリアしました。")
        print("   -> merge_features_final_v4.py を実行してください。")
    else:
        print("⛔ 要確認: 以下の懸念点があります。")
        for i in issues:
            print(f"   - {i}")

if __name__ == "__main__":
    main()