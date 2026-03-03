import os

import pandas as pd

# --- 設定 ---
RAW_DATA_DIR = "1_data/raw/"
PROCESSED_DATA_DIR = "1_data/processed/"
INPUT_STOCK_PRICES = os.path.join(PROCESSED_DATA_DIR, "all_stock_prices.csv")
INDEX_FILES = {
    "is_nikkei_225": "nikkei_225.csv",
    "is_topix_core30": "topix_core30.csv",
    "is_topix_100": "topix_100.csv",
    "is_jpx_nikkei_400": "jpx_nikkei_400.csv",
    "is_growth_core": "growth_core.csv",
    "is_growth_250": "growth_250.csv",
}


def create_index_membership_summary():
    """
    どの銘柄がどの指数に属するかを示すサマリーファイルを作成する
    """
    print("--- 1. 指数所属サマリーを作成中 ---")

    master_list_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, "master_stock_list.csv"))
    master_list_df["code"] = master_list_df["code"].astype(str)
    summary_df = master_list_df.copy()
    index_counts = {}

    for index_name, csv_filename in INDEX_FILES.items():
        file_path = os.path.join(RAW_DATA_DIR, csv_filename)

        if not os.path.exists(file_path):
            print(f"警告: {file_path} が見つかりません。スキップします。")
            continue

        df_index = pd.read_csv(file_path)
        tickers = set(df_index["code"].astype(str))
        summary_df[index_name] = summary_df["code"].apply(lambda x: 1 if x in tickers else 0)
        index_counts[index_name] = len(tickers)

    summary_path = os.path.join(PROCESSED_DATA_DIR, "index_membership_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f" -> 銘柄ごとの指数所属リストを {summary_path} に保存しました。")

    print("\n各指数の銘柄数:")
    for name, count in index_counts.items():
        print(f" - {name}: {count} 銘柄")

    return summary_df


def create_feature_engineering_plan(master_list_df):
    """
    今後の特徴量作成の計画書となるテンプレートファイルを作成する
    """
    print("\n--- 2. 特徴量作成の計画書を作成中 ---")
    plan_df = master_list_df.copy()

    feature_columns = [
        "MovingAverage_5D",
        "MovingAverage_25D",
        "RSI_14D",
        "MACD",
        "SentimentScore_News",
        "TopicVector_1",
        "TopicVector_2",
        "Sales_Growth",
        "PER",
        "PBR",
        "Interest_Rate_JP",
        "USD_JPY_Exchange_Rate",
    ]
    for col in feature_columns:
        plan_df[col] = None

    plan_path = os.path.join(PROCESSED_DATA_DIR, "feature_engineering_plan.csv")
    plan_df.to_csv(plan_path, index=False, encoding="utf-8-sig")
    print(f" -> 特徴量計画のテンプレートを {plan_path} に保存しました。")


def generate_technical_features():
    """
    all_stock_prices.csvを読み込み、基本的なテクニカル指標を追加する
    """
    print("\n--- 3. 基本的なテクニカル特徴量を作成中 ---")
    if not os.path.exists(INPUT_STOCK_PRICES):
        print(f"エラー: {INPUT_STOCK_PRICES} が見つかりません。")
        return

    print(f"{INPUT_STOCK_PRICES} を読み込んでいます...")
    df = pd.read_csv(INPUT_STOCK_PRICES, parse_dates=["Date"], dtype={"Code": str})
    df["Code"] = df["Code"].astype(str)

    print("移動平均線を計算中...")
    df["MA_5D"] = df.groupby("Code")["Close"].transform(lambda x: x.rolling(window=5).mean())
    df["MA_25D"] = df.groupby("Code")["Close"].transform(lambda x: x.rolling(window=25).mean())
    df["MA_75D"] = df.groupby("Code")["Close"].transform(lambda x: x.rolling(window=75).mean())

    df.dropna(inplace=True)

    output_path = os.path.join(PROCESSED_DATA_DIR, "stock_data_with_technical_features.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f" -> テクニカル指標を追加したデータを {output_path} に保存しました。")


if __name__ == "__main__":
    membership_df = create_index_membership_summary()

    if membership_df is not None:
        create_feature_engineering_plan(membership_df[["code", "name"]])

    generate_technical_features()

    print("\n★★★ 全ての処理が完了しました ★★★")
