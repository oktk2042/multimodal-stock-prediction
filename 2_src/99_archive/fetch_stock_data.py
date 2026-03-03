import pandas as pd
import yfinance as yf
import os

# --- 設定 ---
# 銘柄リストTXTが保存されているディレクトリ
TICKER_LIST_DIR = "1_data/raw/" 
# 取得した株価データを保存するディレクトリ
STOCK_PRICES_DIR = "1_data/stock_prices/"
os.makedirs(STOCK_PRICES_DIR, exist_ok=True)

# 取得する期間や間隔
START_DATE = "2018-01-01"
END_DATE = "2025-09-28" # 現在の日付まで取得
INTERVAL = "1d" # "1d" (日次)

# 処理対象のファイルとインデックス名のペア
INDEX_FILES = {
    "nikkei_225_tickers.txt": "Nikkei 225",
    "topix_core30_tickers.txt": "TOPIX Core30",
    "topix_100_tickers.txt": "TOPIX 100",
    "jpx_nikkei_400_tickers.txt": "JPX-Nikkei 400",
    "growth_core_tickers.txt": "TSE Growth Core",
    "growth_250_tickers.txt": "TSE Growth 250"
}

def fetch_all_stock_data():
    """
    銘柄リスト群を元に、yfinanceで株価データを一括取得してインデックスごとに保存する。
    """
    for txt_filename, index_name in INDEX_FILES.items():
        file_path = os.path.join(TICKER_LIST_DIR, txt_filename)
        
        if not os.path.exists(file_path):
            print(f"警告: {file_path} が見つかりません。スキップします。")
            continue

        print(f"--- {index_name} の株価データ取得を開始 ---")
        
        try:
            # .txtファイルからティッカーリストを読み込む
            with open(file_path, 'r') as f:
                # 空行などを除外してリスト化
                tickers = [line.strip() for line in f if line.strip()]
            
            if not tickers:
                print(f" -> {txt_filename} に有効なティッカーがありませんでした。")
                continue

            print(f" -> {len(tickers)}銘柄のデータを一括でダウンロードします...")
            
            # yfinanceで全ティッカーのデータを一括取得
            data = yf.download(
                tickers, 
                start=START_DATE, 
                end=END_DATE, 
                interval=INTERVAL,
                auto_adjust=True,
                group_by='ticker' # 銘柄ごとにデータをグループ化
            )

            if data.empty:
                print(f" -> {index_name} のデータが1件も取得できませんでした。")
                continue
            
            # データを整形して1つのCSVにまとめる
            all_data_list = []
            for ticker in tickers:
                # 取得成功した銘柄のみを処理
                if ticker in data and not data[ticker].dropna().empty:
                    stock_df = data[ticker].copy()
                    stock_df['Ticker'] = ticker
                    # ".T" を削除して元のコードに戻す
                    stock_df['Code'] = ticker.replace('.T', '')
                    all_data_list.append(stock_df)
            
            if not all_data_list:
                print(f" -> {index_name} のデータが整形後、空になりました。")
                continue

            # 全銘柄のデータを結合
            combined_df = pd.concat(all_data_list)
            
            # CSVとして保存
            output_filename = os.path.join(STOCK_PRICES_DIR, f"{index_name.replace(' ', '_').lower()}_stock_prices.csv")
            combined_df.to_csv(output_filename, encoding="utf-8-sig")
            print(f" -> 完了: {len(all_data_list)}銘柄のデータを {output_filename} に保存しました。")

        except Exception as e:
            print(f"エラー: {index_name} の処理中に予期せぬ問題が発生しました: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    fetch_all_stock_data()
    print("\n★★★ すべての株価データ取得処理が完了しました。 ★★★")
