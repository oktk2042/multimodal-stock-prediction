import pandas as pd
from pathlib import Path
from tqdm import tqdm
import collections

# --- 設定 ---
BASE_DIR = Path("./1_data/edinet_reports/")
# 入力元：解凍されたファイルが保存されている親フォルダ
UNZIPPED_SOURCE_DIR = BASE_DIR / "02_unzipped_files/"
# 最終成果物：分析結果を保存するCSVファイルのパス
OUTPUT_ANALYSIS_CSV_PATH = BASE_DIR / "csv_structure_analysis.csv"


def analyze_all_csv_files():
    """
    解凍済みの全フォルダを探索し、内部の全CSVファイルの構造を分析して
    項目名の出現頻度などを集計したサマリーファイルを出力する。
    """
    print(f"--- {UNZIPPED_SOURCE_DIR} 内の全CSVファイルの構造分析を開始します ---")
    
    # 全てのCSVファイルのパスを取得
    all_csv_files = list(UNZIPPED_SOURCE_DIR.rglob('*.csv'))
    if not all_csv_files:
        print(f"エラー: {UNZIPPED_SOURCE_DIR} 内にCSVファイルが見つかりません。")
        return

    print(f"{len(all_csv_files)}個のCSVファイルを分析します。")

    # 全ての行の情報を格納するリスト
    all_rows_data = []

    for csv_path in tqdm(all_csv_files, desc="CSVファイル分析中"):
        try:
            # utf-16-leで読み込み、ヘッダーとして1行目を指定
            df = pd.read_csv(csv_path, sep='\t', header=0, encoding='utf-16-le', on_bad_lines='skip')
            
            # DocIDをファイル名から取得
            doc_id = csv_path.parts[-3] # .../S100XXXX/XBRL_TO_CSV/file.csv -> S100XXXX
            df['DocID'] = doc_id
            
            all_rows_data.append(df)

        except Exception as e:
            tqdm.write(f"ファイル読み込みエラー: {csv_path}, 詳細: {e}")

    if not all_rows_data:
        print("分析できるデータがありませんでした。")
        return

    # 全てのDataFrameを一つに結合
    print("\n全データを結合中...")
    master_df = pd.concat(all_rows_data, ignore_index=True)

    print("項目名の出現頻度を集計中...")
    # 「項目名」列の出現回数をカウント
    item_name_counts = master_df['項目名'].value_counts().reset_index()
    item_name_counts.columns = ['項目名', '出現回数']
    
    print("代表的なコンテキストIDを集計中...")
    # 各項目名に対して、最もよく使われるコンテキストIDを一つ取得
    # drop_duplicatesで各項目名の最初の行を取得し、必要な列だけを選択
    representative_contexts = master_df.drop_duplicates(subset=['項目名'])[['項目名', 'コンテキストID', '連結・個別', '期間・時点', '単位']]
    
    # 頻度データと代表コンテキストをマージ
    analysis_summary = pd.merge(item_name_counts, representative_contexts, on='項目名', how='left')

    # 結果をCSVに保存
    analysis_summary.to_csv(OUTPUT_ANALYSIS_CSV_PATH, index=False, encoding="utf-8-sig")

    print("\n★★★ 構造分析が完了しました ★★★")
    print(f"分析結果を {OUTPUT_ANALYSIS_CSV_PATH} に保存しました。")
    print("\n上位30項目:")
    print(analysis_summary.head(30).to_string())


if __name__ == "__main__":
    analyze_all_csv_files()