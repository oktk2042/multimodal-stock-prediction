import pandas as pd
import os
from pathlib import Path

# ==========================================
# 設定
# ==========================================
# プロジェクトのルートディレクトリ
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル
NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"

# 出力先ディレクトリ
OUTPUT_NEWS_DIR = DATA_DIR / "news_by_code"
OUTPUT_PRICE_DIR = DATA_DIR / "price_by_code"

def split_csv_by_code(input, output):
    # 出力ディレクトリが存在しない場合は作成
    output.mkdir(parents=True, exist_ok=True)
    
    print(f"読み込み中: {input}")
    
    try:
        # CSVファイルの読み込み
        # データ量が多い場合は low_memory=False または dtype指定を推奨
        df = pd.read_csv(input, dtype={'code': str})
        
        # 日付型への変換（念のため）
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])

        # ユニークな銘柄コードを取得
        unique_codes = df['code'].unique()
        print(f"ユニークな銘柄数: {len(unique_codes)}")
        print(f"出力先フォルダ: {output}")

        # コードごとに分割して保存
        count = 0
        for code in unique_codes:
            # その銘柄のデータのみ抽出
            df_code = df[df['code'] == code].copy()
            
            # 日付順にソート（時系列分析に使いやすくするため）
            df_code = df_code.sort_values('Date')
            
            # 保存ファイル名
            output_path = output / f"price_{code}.csv"
            
            # CSV保存
            df_code.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            count += 1
            if count % 100 == 0:
                print(f"  {count} 銘柄処理完了...")

        print(f"\n完了しました。合計 {count} 個のファイルを作成しました。")

    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {input}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

if __name__ == "__main__":
    split_csv_by_code(PRICE_FILE, OUTPUT_PRICE_DIR)
    # split_csv_by_code(NEWS_FILE, OUTPUT_NEWS_DIR)