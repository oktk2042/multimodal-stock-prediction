import os

import pandas as pd

# --- 設定 ---
PROCESSED_DATA_DIR = "1_data/processed/"
INPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "stock_data_with_technical_features.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "stock_data_with_all_technicals.csv")


def add_advanced_technical_features():
    """
    既存の特徴量データに、pandas_taを使ってRSIとMACDなどを追加する。
    """
    print("--- 高度なテクニカル特徴量の作成を開始 ---")
    if not os.path.exists(INPUT_FILE):
        print(f"エラー: {INPUT_FILE} が見つかりません。")
        return

    print(f"{INPUT_FILE} を読み込んでいます...")
    df = pd.read_csv(INPUT_FILE, parse_dates=["Date"], dtype={"Code": str})

    # pandas_taを使いやすくするため、列名を小文字にリネーム
    df.rename(
        columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True
    )

    print("銘柄ごとにRSI, MACD, ボリンジャーバンドを計算中...")

    # 各テクニカル指標を個別に追加する関数を定義
    def calculate_indicators(group):
        # RSIを追加 (dfに 'RSI_14' という列が追加される)
        group.ta.rsi(append=True)
        # MACDを追加 (dfに 'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9' が追加される)
        group.ta.macd(append=True)
        # ボリンジャーバンドを追加 (dfに 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', ... が追加される)
        group.ta.bbands(length=20, append=True)
        return group

    # 銘柄ごと(Code)にグループ化し、定義した関数を適用
    df = df.groupby("Code", group_keys=False).apply(calculate_indicators)

    # 元の列名に戻す
    df.rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True
    )

    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f" -> RSI, MACD等を追加したデータを {OUTPUT_FILE} に保存しました。")


if __name__ == "__main__":
    add_advanced_technical_features()
    print("\n★★★ 処理が完了しました ★★★")
