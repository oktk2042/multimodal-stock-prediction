import pandas as pd
import os

# --- 設定 ---
SOURCE_DATA_DIR = "1_data/raw/" 
SAVE_DIR = "1_data/raw/"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_FILES = [
    "nikkei_225.csv",
    "topix_core30.csv",
    "topix_100.csv",
    "jpx_nikkei_400.csv",
    "growth_core.csv",
    "growth_250.csv"
]

def extract_tickers_from_csv_files():
    """
    複数のCSVファイルから銘柄コードを読み込み、
    それぞれに対応するティッカーリスト(.txt)を作成する。
    """
    print("--- CSVからのティッカーリスト作成を開始 ---")

    for filename in CSV_FILES:
        input_path = os.path.join(SOURCE_DATA_DIR, filename)
        
        if not os.path.exists(input_path):
            print(f"警告: {input_path} が見つかりません。スキップします。")
            continue
        
        try:
            print(f" -> {filename} を処理中...")
            df = pd.read_csv(input_path, encoding='utf-8')
            
            if 'code' not in df.columns:
                print(f"警告: {filename} に 'code' カラムがありません。スキップします。")
                continue

            tickers = df['code'].astype(str).tolist()
            tickers_with_suffix = [f"{ticker}.T" for ticker in tickers]
            output_filename = filename.replace('.csv', '_tickers.txt')
            save_path = os.path.join(SAVE_DIR, output_filename)
            
            with open(save_path, 'w') as f:
                f.write('\n'.join(tickers_with_suffix))
            
            print(f" -> {len(tickers_with_suffix)} 銘柄を {save_path} に保存しました。")

        except Exception as e:
            print(f"エラー: {filename} の処理中に問題が発生しました - {e}")

    print("\n--- すべてのティッカーリストの作成が完了しました。 ---")

if __name__ == "__main__":
    extract_tickers_from_csv_files()