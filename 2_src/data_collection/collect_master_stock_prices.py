import os

import pandas as pd
import yfinance as yf

# --- 設定 ---
# マスターリストCSVのパス
MASTER_LIST_PATH = "1_data/processed/master_stock_list.csv"
# 取得した全株価データを保存するパス
SAVE_PATH = "1_data/processed/all_stock_prices.csv"

# 取得する期間や間隔
START_DATE = "2018-01-01"
# JSTでの現在の日付を取得
END_DATE = pd.to_datetime("today", utc=True).tz_convert("Asia/Tokyo").strftime("%Y-%m-%d")
INTERVAL = "1d"  # "1d" (日次)


def fetch_master_stock_prices():
    """
    マスターリストに基づき、全銘柄の株価データを一括で取得して保存する。
    """
    if not os.path.exists(MASTER_LIST_PATH):
        print(f"エラー: {MASTER_LIST_PATH} が見つかりません。")
        return

    print("--- 全銘柄の株価データ収集を開始します ---")

    # マスターリストを読み込む
    master_df = pd.read_csv(MASTER_LIST_PATH)

    # yfinanceで使えるように、証券コードに ".T" を付加
    tickers = [f"{code}.T" for code in master_df["code"]]

    print(f"対象銘柄数: {len(tickers)} 件")
    print(f"期間: {START_DATE} から {END_DATE} まで")
    print("ダウンロードを開始します。銘柄数が多いため、数分かかる場合があります...")

    try:
        # yfinanceで全ティッカーのデータを一括取得
        # データはワイド形式（列がマルチレベルインデックス）で返される
        wide_data = yf.download(tickers, start=START_DATE, end=END_DATE, interval=INTERVAL, auto_adjust=True)

        if wide_data.empty:
            print("エラー: データを1件も取得できませんでした。")
            return

        print("\nダウンロード完了。データを整形します...")

        # ワイド形式からロング形式へデータを変換
        # stack()を使って、銘柄ごとの列を行に変換する
        long_data = wide_data.stack(level=1).reset_index()
        # 列名を分かりやすく変更
        long_data.rename(columns={"level_1": "Ticker"}, inplace=True)

        # 銘柄コード(数字のみ)の列を追加
        long_data["Code"] = long_data["Ticker"].str.replace(".T", "", regex=False)

        # 銘柄名の列を追加するために、マスターリストとマージ
        # 'code'列の型を文字列に統一
        master_df["code"] = master_df["code"].astype(str)
        final_df = pd.merge(long_data, master_df, left_on="Code", right_on="code", how="left")

        # 不要な列を削除し、列順を整理
        final_df = final_df[["Date", "Code", "name", "Open", "High", "Low", "Close", "Volume"]]
        final_df.rename(columns={"name": "Name"}, inplace=True)

        # CSVとして保存
        final_df.to_csv(SAVE_PATH, index=False, encoding="utf-8-sig")
        print(f"\n--- 全株価データを {SAVE_PATH} に保存しました。 ---")
        print(f"合計 {len(final_df['Code'].unique())} 銘柄、{len(final_df)}行のデータを保存。")

    except Exception as e:
        print(f"\nエラー: 処理中に予期せぬ問題が発生しました: {e}")


if __name__ == "__main__":
    fetch_master_stock_prices()
