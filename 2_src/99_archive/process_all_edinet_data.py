import time
import shutil
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser

# --- 設定 ---
BASE_DIR = Path("./1_data/edinet_reports/")
ZIP_SOURCE_DIR = BASE_DIR / "01_zip_files/"
UNZIPPED_DEST_DIR = BASE_DIR / "02_unzipped_files/"
OUTPUT_CSV_PATH = BASE_DIR / "edinet_financial_data.csv"

# 抽出したい財務項目のキーワード（CSVの1列目に含まれる文字列）
# 必要に応じて、このリストにキーワードを追加・編集してください
TARGET_KEYWORDS = {
    # 出力列名: [CSV内で検索するキーワード候補（優先順位順）]
    'NetSales': ['売上収益（IFRS）', '売上高'],
    'OperatingIncome': ['営業利益'],
    'NetIncome': ['親会社の所有者に帰属する当期利益', '当期純利益'],
    'TotalAssets': ['資産合計', '総資産額', '総資産'],
    'NetAssets': ['親会社の所有者に帰属する持分', '純資産額', '純資産'],
    'EquityToAssetRatio': ['親会社所有者帰属持分比率', '自己資本比率'],
    'NumberOfIssuedShares': ['発行済株式総数'] # 「株式の総数」は発行可能株式数も含むため除外
}

# ==============================================================================
# ステップ1：全ZIPファイル解凍関数
# ==============================================================================
def unzip_all_files():
    """
    ZIP_SOURCE_DIR内の全てのZIPファイルを、UNZIPPED_DEST_DIRに解凍する。
    """
    zip_files = list(ZIP_SOURCE_DIR.rglob('*.zip'))
    if not zip_files:
        print(f"エラー: {ZIP_SOURCE_DIR} 内に処理対象のZIPファイルが見つかりません。")
        return False

    print(f"--- ステップ1：{len(zip_files)}個のZIPファイルを {UNZIPPED_DEST_DIR} に解凍します ---")
    UNZIPPED_DEST_DIR.mkdir(exist_ok=True)

    for zip_path in tqdm(zip_files, desc="ZIPファイル解凍中"):
        doc_id = zip_path.stem
        target_dir = UNZIPPED_DEST_DIR / doc_id
        if target_dir.exists():
            continue
        target_dir.mkdir()
        try:
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(target_dir)
        except Exception as e:
            tqdm.write(f"解凍エラー: {zip_path}, 詳細: {e}")
            shutil.rmtree(target_dir)
        
        # 各ファイルの処理の間に0.1秒の待機時間を追加
        time.sleep(0.1)

    print("\n★★★ 全てのファイルの解凍が完了しました ★★★\n")
    return True


# ==============================================================================
# ステップ2：解凍済みファイル解析関数
# ==============================================================================
def parse_single_csv(csv_path: Path, doc_id: str) -> dict:
    """単一のCSVファイルを読み込み、TARGET_KEYWORDSに一致する項目を抽出する。"""
    record = {"DocID": doc_id}
    try:
        df = pd.read_csv(csv_path, sep='\t', header=0, index_col=0, 
                         on_bad_lines='skip', encoding='utf-16-le', low_memory=False)
        
        # 項目名列(df.iloc[:, 0])と値列(df.iloc[:, -1])が存在するか確認
        if df.shape[1] < 2:
            return {}
            
        item_names = df.iloc[:, 0]
        values = df.iloc[:, -1] # 常に最後の列を値として取得

        for label, keywords in TARGET_KEYWORDS.items():
            found_value = None
            for keyword in keywords:
                # キーワードに部分一致する行を探す
                matched_mask = item_names.str.contains(keyword, na=False)
                if matched_mask.any():
                    # 最初にヒットした行の値を取得
                    found_value = values[matched_mask].iloc[0]
                    break # 優先順位の高いキーワードで見つかったらループを抜ける
            record[label] = found_value
            
    except Exception as e:
        tqdm.write(f"CSV解析エラー: {csv_path.name}, 詳細: {e}")
        
    return record

def parse_all_csv_files_in_folders():
    """解凍済みの全フォルダを処理し、内部の全CSVから財務データを抽出して単一のCSVに保存する"""
    unzipped_folders = [d for d in UNZIPPED_DEST_DIR.iterdir() if d.is_dir()]
    if not unzipped_folders:
        print(f"エラー: {UNZIPPED_DEST_DIR} 内に処理対象のフォルダが見つかりません。")
        return

    print(f"--- {len(unzipped_folders)}個の解凍済みフォルダ内の全CSVを解析します ---")
    
    all_records = []
    for folder_path in tqdm(unzipped_folders, desc="フォルダ解析中"):
        doc_id = folder_path.name
        csv_files = list(folder_path.rglob('*.csv'))
        if not csv_files:
            continue

        combined_data_for_doc = {}
        for csv_path in csv_files:
            extracted_data = parse_single_csv(csv_path, doc_id)
            # まだ値が見つかっていない項目のみ更新
            for key, value in extracted_data.items():
                if key not in combined_data_for_doc or pd.isna(combined_data_for_doc[key]):
                    if pd.notna(value):
                        combined_data_for_doc[key] = value

        # DocIDを最初に追加
        final_record = {"DocID": doc_id}
        final_record.update(combined_data_for_doc)

        # 何か一つでもデータが抽出できたら記録（DocID以外のキーで）
        if any(pd.notna(v) for k, v in final_record.items() if k != 'DocID'):
            all_records.append(final_record)

    if all_records:
        summary_df = pd.DataFrame(all_records)
        summary_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding="utf-8-sig")
        print("\n★★★ 処理完了 ★★★")
        print(f"{len(summary_df)}件分の財務データを {OUTPUT_CSV_PATH} に保存しました。")
    else:
        print("\n--- データを抽出できるCSVファイルがありませんでした ---")

# ==============================================================================
# メイン処理
# ==============================================================================
def main():
    """メイン処理を実行する"""
    unzip_success = unzip_all_files()
    if unzip_success:
        parse_all_csv_files_in_folders()
    print("\n全ての処理が完了しました。🎉")

if __name__ == "__main__":
    main()