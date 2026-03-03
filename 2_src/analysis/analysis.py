from pathlib import Path

import pandas as pd

# --- 設定 ---
base_path = Path("C:/M2_Research_Project/1_data")
raw_path = base_path / "raw"
processed_path = base_path / "processed"

# 入力ファイル
stock_features_file = processed_path / "index_membership_summary.csv"
ticker_file = raw_path / "top_200_trad_val_tickers_filtered.txt"

try:
    # --- ファイルの読み込み ---
    # 全銘柄の指数構成データ
    df_all = pd.read_csv(stock_features_file)

    # TOP200銘柄のティッカーリスト
    with open(ticker_file) as f:
        top_200_tickers_raw = [line.strip() for line in f.readlines()]

    # --- データの前処理 ---
    # ご指摘に基づき、TOP200のティッカーを文字列のまま保持
    top_200_codes = [ticker.replace(".T", "") for ticker in top_200_tickers_raw]

    # 対象となる指数カラムの名前をリスト化
    index_columns = [
        "is_nikkei_225",
        "is_topix_core30",
        "is_topix_100",
        "is_jpx_nikkei_400",
        "is_growth_core",
        "is_growth_250",
    ]

    # --- 集計処理 ---
    # 1. 全銘柄における各指数の銘柄数を計算
    all_stocks_counts = df_all[index_columns].sum().rename("全銘柄")

    # 2. 全銘柄リストをTOP200の銘柄コードでフィルタリング
    df_top_200 = df_all[df_all["code"].isin(top_200_codes)]

    # 3. TOP200銘柄における各指数の銘柄数を計算
    top_200_stocks_counts = df_top_200[index_columns].sum().rename("TOP200銘柄")

    # --- 結果の整形と出力 ---
    # 全銘柄とTOP200銘柄の集計結果を一つのデータフレームに結合
    summary_df = pd.concat([all_stocks_counts, top_200_stocks_counts], axis=1)

    # 各指数の名前を日本語に変換して見やすくする
    summary_df.index = ["日経225", "TOPIX Core30", "TOPIX 100", "JPX日経400", "グロース・コア", "グロース250"]

    # 結果をCSVファイルとして出力
    output_filename = "index_composition_summary.csv"
    summary_df.to_csv(output_filename, encoding="utf-8-sig")

    print(f"'{output_filename}' を作成しました。")
    print("\n--- 分析結果 ---")
    print("各指数に含まれる全銘柄数と、そのうちTOP200に選出された銘柄数:")
    print(summary_df)

except FileNotFoundError as e:
    print("エラー: ファイルが見つかりません。")
    print(f"'{e.filename}' がPythonスクリプトと同じフォルダにあるか確認してください。")
except Exception as e:
    print(f"予期せぬエラーが発生しました: {e}")
