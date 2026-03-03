import pandas as pd
import os
from pathlib import Path

# --- 設定 ---
BASE_DIR = Path("C:/M2_Research_Project")
PROCESSED_DIR = BASE_DIR / "1_data" / "processed"
RAW_DIR = BASE_DIR / "1_data" / "raw"
INPUT_CSV_PATH = PROCESSED_DIR / "stock_data_features_v1.csv"
OUTPUT_TXT_PATH = RAW_DIR / "top_200_trad_val_tickers_filtered.txt" # 出力ファイル名を変更

# --- フィルタリング条件 ---
NUM_TOP_STOCKS = 200      # 最終的に抽出する上位銘柄数
START_DATE = "2020-01-01" # 分析対象の開始日
END_DATE = "2025-01-01"   # 分析対象の終了日
# この期間の約80%以上の営業日データを持つことを基準とする (5年 x 約245日 x 80% ~= 980)
MIN_RECORDS_THRESHOLD = 980 

def filter_top_stocks_by_quality_and_value():
    """
    指定された期間内に十分なデータレコードを持つ銘柄をまず絞り込み、
    その中からさらに平均売買代金上位の銘柄リストをテキストファイルとして出力する。
    """
    print(f"--- 特徴量ファイル {INPUT_CSV_PATH.name} を読み込みます ---")
    try:
        df = pd.read_csv(
            INPUT_CSV_PATH, 
            dtype={'Code': str}, # 証券コードを文字列として読み込む
            parse_dates=['Date'] # Date列を日付型として読み込む
        )
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {INPUT_CSV_PATH}")
        return

    # --- 1.【品質・期間フィルタ】 ---
    print(f"\n--- ステップ1: 品質と期間によるフィルタリング ---")
    print(f"分析対象期間: {START_DATE} から {END_DATE}")
    
    # 指定期間内のデータに絞り込む
    df_period = df[(df['Date'] >= START_DATE) & (df['Date'] <= END_DATE)].copy()
    
    # 銘柄ごとにレコード数をカウント
    record_counts = df_period.groupby('Code').size()
    
    # レコード数が閾値以上の銘柄コードをリストアップ
    eligible_codes = record_counts[record_counts >= MIN_RECORDS_THRESHOLD].index.tolist()
    
    print(f"{len(eligible_codes)}銘柄が、指定期間内に十分なデータ（{MIN_RECORDS_THRESHOLD}日以上）を持っていることが確認されました。")
    
    # 十分なデータを持つ銘柄のデータのみを抽出
    df_eligible = df_period[df_period['Code'].isin(eligible_codes)].copy()
    
    # --- 2.【売買代金フィルタ】 ---
    print(f"\n--- ステップ2: 売買代金によるフィルタリング ---")
    print("日々の売買代金を計算しています...")
    df_eligible['TradingValue'] = df_eligible['Close'] * df_eligible['Volume']

    print("対象銘柄の平均売買代金を計算しています...")
    avg_trading_value = df_eligible.groupby('Code')['TradingValue'].mean()

    print(f"平均売買代金の上位{NUM_TOP_STOCKS}銘柄を抽出しています...")
    top_stocks_codes = avg_trading_value.sort_values(ascending=False).head(NUM_TOP_STOCKS).index.tolist()

    # --- 3. 結果の保存 ---
    RAW_DIR.mkdir(exist_ok=True)
    print(f"最終的なティッカーリストを {OUTPUT_TXT_PATH} に保存しています...")
    with open(OUTPUT_TXT_PATH, 'w', encoding='utf-8') as f:
        for code in top_stocks_codes:
            f.write(f"{code}.T\n")
            
    print("\n★★★ 高品質な銘柄の絞り込み処理が完了しました ★★★")
    print(f"今後は {OUTPUT_TXT_PATH.name} を分析対象の銘柄リストとして使用してください。")

if __name__ == "__main__":
    filter_top_stocks_by_quality_and_value()
