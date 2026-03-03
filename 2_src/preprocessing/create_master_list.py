import os

import pandas as pd

# --- 設定 ---
TICKER_LIST_DIR = "1_data/raw/"
PROCESSED_DATA_DIR = "1_data/processed/"
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

CSV_FILES = [
    "nikkei_225.csv",
    "topix_core30.csv",
    "topix_100.csv",
    "jpx_nikkei_400.csv",
    "growth_core.csv",
    "growth_250.csv",
]


def create_master_stock_list():
    all_stocks_list = []
    print("--- 全銘柄マスターリストの作成を開始 ---")

    for filename in CSV_FILES:
        file_path = os.path.join(TICKER_LIST_DIR, filename)

        if not os.path.exists(file_path):
            print(f"警告: {file_path} が見つかりません。スキップします。")
            continue

        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            if "code" in df.columns and "name" in df.columns:
                all_stocks_list.append(df[["code", "name"]])
                print(f" -> {filename} から {len(df)} 銘柄を読み込みました。")
            else:
                print(f"警告: {filename} に 'code' または 'name' カラムがありません。")
        except Exception as e:
            print(f"エラー: {filename} の読み込み中に問題が発生しました - {e}")

    if not all_stocks_list:
        print("読み込める銘柄リストがありませんでした。処理を終了します。")
        return

    master_df = pd.concat(all_stocks_list, ignore_index=True)
    print(f"\n結合後の総銘柄数（重複あり）: {len(master_df)}")
    master_df["code"] = master_df["code"].astype(str)
    master_df.drop_duplicates(subset=["code"], keep="first", inplace=True)
    print(f"重複除去後のユニーク銘柄数: {len(master_df)}")
    master_df.sort_values(by="code", inplace=True)
    output_path = os.path.join(PROCESSED_DATA_DIR, "master_stock_list.csv")
    master_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n--- 全銘柄マスターリストを {output_path} に保存しました。 ---")


if __name__ == "__main__":
    create_master_stock_list()
