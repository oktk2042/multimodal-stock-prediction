import pandas as pd
from pathlib import Path

def preprocess_daily_stock_data():
    """
    巨大な日次株価データから、分析対象の銘柄と期間を絞り込み、
    モデルが使いやすい形の前処理済みデータセットを作成する。
    """
    try:
        # --- 1. ファイルパスとパラメータの設定 ---
        base_path = Path("C:/M2_Research_Project/1_data")
        raw_path = base_path / "raw"
        processed_path = base_path / "processed"

        # 入力ファイル
        stock_features_file = processed_path / "stock_data_features_v1.csv"
        ticker_file = raw_path / "top_200_trad_val_tickers_filtered.txt"

        # 出力ファイル
        output_file = processed_path / "preprocessed_top200_daily_data.csv"

        # パラメータ設定
        START_DATE = "2020-01-01"
        END_DATE = "2025-01-01" # データに合わせて調整してください

        print("--- 前処理を開始します ---")

        # --- 2. 分析対象の200銘柄リストを読み込み ---
        with open(ticker_file, 'r', encoding='utf-8') as f:
            # .T を除いた証券コードのリストを作成
            target_codes = [line.strip().replace('.T', '') for line in f]
        
        print(f"対象銘柄数: {len(target_codes)}")

        # --- 3. 日次株価データを読み込み ---
        print(f"'{stock_features_file.name}' を読み込んでいます... (時間がかかる場合があります)")
        stock_df = pd.read_csv(
            stock_features_file,
            parse_dates=['Date'],
            dtype={'Code': str} # 証券コードを文字列として読み込む
        )
        print("読み込み完了。")

        # --- 4. 銘柄と期間でデータを絞り込み ---
        # 銘柄で絞り込み
        filtered_df = stock_df[stock_df['Code'].isin(target_codes)].copy()
        print(f"200銘柄に絞り込みました。データ数: {len(filtered_df)} 行")
        
        # 期間で絞り込み
        filtered_df = filtered_df[(filtered_df['Date'] >= START_DATE) & (filtered_df['Date'] <= END_DATE)]
        print(f"{START_DATE} から {END_DATE} の期間に絞り込みました。データ数: {len(filtered_df)} 行")

        # --- 5. 結果をCSVファイルに保存 ---
        filtered_df.sort_values(by=['Code', 'Date'], inplace=True)
        filtered_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n--- 前処理完了 ---")
        print(f"処理済みのデータが '{output_file}' に保存されました。")
        
        return filtered_df

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりませんでした。パスを確認してください: {e.filename}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None

if __name__ == '__main__':
    preprocessed_df = preprocess_daily_stock_data()
    if preprocessed_df is not None:
        print("\n--- 生成されたデータの先頭5行 ---")
        print(preprocessed_df.head())
        print("\n--- 生成されたデータの末尾5行 ---")
        print(preprocessed_df.tail())
