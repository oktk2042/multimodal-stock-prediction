import pandas as pd
import os

# --- 設定 ---
PROCESSED_DATA_DIR = "1_data/processed/"
RAW_DATA_DIR = "1_data/raw/"

# --- 入力ファイル ---
STOCK_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "stock_data_with_all_technicals.csv")
EXCHANGE_RATE_FILE = os.path.join(RAW_DATA_DIR, "usd_jpy_exchange_rate.csv")
INDEX_MEMBERSHIP_FILE = os.path.join(PROCESSED_DATA_DIR, "index_membership_summary.csv")

# --- 出力ファイル ---
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "stock_data_features_v1.csv")

def merge_all_features():
    print("--- 全特徴量のマージ処理を開始 ---")

    print("各データファイルを読み込んでいます...")
    try:
        df_stocks = pd.read_csv(STOCK_DATA_FILE, parse_dates=['Date'], dtype={'Code': str})
        df_exchange = pd.read_csv(
            EXCHANGE_RATE_FILE,
            skiprows=3, # 先頭の3行を無視する
            names=['Date', 'USD_JPY'] # 列名を直接指定
        )
        df_exchange['Date'] = pd.to_datetime(df_exchange['Date'])
        df_membership = pd.read_csv(INDEX_MEMBERSHIP_FILE, dtype={'code': str})

    except FileNotFoundError as e:
        print(f"エラー: 入力ファイルが見つかりません。 {e}")
        return

    print("ステップ1: 株価データに為替レートをマージします...")
    df_merged = pd.merge(df_stocks, df_exchange, on='Date', how='left')
    df_merged['USD_JPY'] = df_merged['USD_JPY'].ffill()
    df_merged['USD_JPY'] = df_merged['USD_JPY'].bfill()
    print("ステップ2: 指数所属情報をマージします...")
    df_membership.rename(columns={'code': 'Code'}, inplace=True)
    df_final = pd.merge(df_merged, df_membership.drop(columns=['name']), on='Code', how='left')
    index_flag_columns = [col for col in df_final.columns if col.startswith('is_')]
    df_final[index_flag_columns] = df_final[index_flag_columns].fillna(0).astype(int)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n--- 完了: 全ての特徴量を追加した最終版データを {OUTPUT_FILE} に保存しました。 ---")
    print(f"最終的な特徴量カラム数: {len(df_final.columns)} 件")


if __name__ == "__main__":
    merge_all_features()