import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

FILES = {
    "Stock_Prices": DATA_DIR / "stock_data_features_v1.csv",
    "Financials_LLM": DATA_DIR / "edinet_features_llm.csv",
    "Financials_Hybrid": DATA_DIR / "edinet_features_financials_hybrid.csv",
    "Financials_FinBERT": DATA_DIR / "edinet_features_finbert.csv",
    "News_Scores": DATA_DIR / "news_sentiment_historical.csv",
    "Sector_Info": DATA_DIR / "stock_sector_info.csv",
    "Global_Macro": DATA_DIR / "global_macro_features.csv"
}

def analyze_file(name, path):
    print(f"\n=== {name} ===")
    if not path.exists():
        print("× ファイルが存在しません")
        return
    
    try:
        # 巨大ファイル対策
        if "News" in name:
            df = pd.read_csv(path, nrows=5000, encoding='utf-8-sig')
            print("(※巨大ファイルのため先頭5000行のみチェック)")
        else:
            df = pd.read_csv(path, encoding='utf-8-sig')
            
        print(f"行数: {len(df):,} | カラム数: {len(df.columns)}")
        print(f"カラム一覧: {list(df.columns)}")
        
        # 欠損チェック
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            print("欠損ありカラム:")
            print(missing)
        else:
            print("欠損なし (完全)")
            
    except Exception as e:
        print(f"エラー: {e}")

def main():
    print("--- 特徴量インベントリ分析 ---")
    for name, path in FILES.items():
        analyze_file(name, path)

if __name__ == "__main__":
    main()