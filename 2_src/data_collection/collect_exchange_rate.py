import os

import pandas as pd
import yfinance as yf

# --- 設定 ---
SAVE_DIR = "1_data/raw/"
START_DATE = "2018-01-01"
END_DATE = pd.to_datetime("today", utc=True).tz_convert("Asia/Tokyo").strftime("%Y-%m-%d")


def collect_usd_jpy_rate():
    """
    USD/JPYの為替レートデータをyfinanceで取得する
    """
    print("--- USD/JPY 為替レートデータを取得中 ---")
    try:
        # "JPY=X" がUSD/JPYのティッカーシンボル
        data = yf.download("JPY=X", start=START_DATE, end=END_DATE, auto_adjust=True)

        if data.empty:
            print("エラー: 為替レートデータが取得できませんでした。")
            return

        # 終値だけを残し、列名を'USD_JPY'に変更
        df_rate = data[["Close"]].rename(columns={"Close": "USD_JPY"})

        save_path = os.path.join(SAVE_DIR, "usd_jpy_exchange_rate.csv")
        df_rate.to_csv(save_path, encoding="utf-8-sig")
        print(f" -> 為替レートデータを {save_path} に保存しました。")

    except Exception as e:
        print(f"エラー: 処理中に問題が発生しました - {e}")


if __name__ == "__main__":
    collect_usd_jpy_rate()
