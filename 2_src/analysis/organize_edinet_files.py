import shutil
from pathlib import Path

import pandas as pd

# ==========================================
# 1. 設定 (Configuration)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 入力ファイル・フォルダ
SUMMARY_FILE = PROJECT_ROOT / "1_data" / "processed" / "EDINET_Summary_v3.csv"
UNZIPPED_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

# 出力先フォルダ（銘柄コードごとに整理される場所）
OUTPUT_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "03_organized_by_code"

# 動作モード
# True: 実際にはコピーせず、ログ出力のみ（テスト用）
# False: 実際にファイルをコピーする
DRY_RUN = False


def load_docid_mapping(summary_file):
    """EDINET SummaryからdocIDと銘柄コードのマッピングを作成する"""
    print(f"Loading summary file: {summary_file}")
    try:
        # 必要なカラムのみ読み込み
        df = pd.read_csv(summary_file, usecols=["docID", "secCode", "filerName", "submitDateTime"])

        # secCodeがNaNの行（ファンドなど）を除外
        df = df.dropna(subset=["secCode"])

        # secCodeを整数型に変換し、末尾の0を削除して4桁にする
        # 例: 75390 -> 7539
        df["Code"] = df["secCode"].astype(str).str[:-1]

        # マッピング辞書の作成 {docID: Code}
        mapping = dict(zip(df["docID"], df["Code"]))

        print(f"Mapping created: {len(mapping)} documents linked to codes.")
        return mapping, df
    except Exception as e:
        print(f"Error loading summary file: {e}")
        return {}, pd.DataFrame()


def organize_files(mapping):
    """ファイルを銘柄コードごとのフォルダに整理する"""
    print(f"Scanning directory: {UNZIPPED_DIR}")

    # S100* フォルダを走査
    # Path.glob はジェネレータなのでメモリ効率が良い
    doc_dirs = list(UNZIPPED_DIR.glob("S100*"))
    print(f"Found {len(doc_dirs)} document directories.")

    count_success = 0
    count_skip = 0
    count_error = 0

    for doc_dir in doc_dirs:
        doc_id = doc_dir.name

        # マッピングに存在しないdocID（上場企業以外など）はスキップ
        if doc_id not in mapping:
            # print(f"Skipping {doc_id}: No code mapping found.")
            count_skip += 1
            continue

        code = mapping[doc_id]

        # XBRL_TO_CSV フォルダ内のCSVを探す
        csv_dir = doc_dir / "XBRL_TO_CSV"
        if not csv_dir.exists():
            continue

        csv_files = list(csv_dir.glob("*.csv"))

        if not csv_files:
            continue

        # 出力先ディレクトリの作成
        target_dir = OUTPUT_DIR / code
        if not DRY_RUN:
            target_dir.mkdir(parents=True, exist_ok=True)

        for csv_file in csv_files:
            # ファイル名を変更してコピーする場合
            # 例: S100JF5C_jpaud-qrr-cc-001...csv
            new_filename = f"{doc_id}_{csv_file.name}"
            target_path = target_dir / new_filename

            try:
                if DRY_RUN:
                    print(f"[Dry Run] Copy {csv_file} -> {target_path}")
                else:
                    # コピー実行 (メタデータもコピー)
                    shutil.copy2(csv_file, target_path)
                count_success += 1
            except Exception as e:
                print(f"Error copying {csv_file}: {e}")
                count_error += 1

        if count_success % 100 == 0:
            print(f"Processed {count_success} files...")

    print("\n--- Summary ---")
    print(f"Files Processed (Success): {count_success}")
    print(f"Skipped directories (No Mapping): {count_skip}")
    print(f"Errors: {count_error}")
    print(f"Output Directory: {OUTPUT_DIR}")


def main():
    # 1. マッピングの読み込み
    if not SUMMARY_FILE.exists():
        print(f"Error: Summary file not found at {SUMMARY_FILE}")
        # ダミーデータでの動作確認用（実際には不要）
        # mapping = {'S100JF5C': '4689', 'S100JRD2': '8227'}
        return

    mapping, _ = load_docid_mapping(SUMMARY_FILE)

    if not mapping:
        print("No mapping data available. Exiting.")
        return

    # 2. ファイルの整理実行
    if not UNZIPPED_DIR.exists():
        print(f"Error: Unzipped directory not found at {UNZIPPED_DIR}")
        return

    organize_files(mapping)


if __name__ == "__main__":
    main()
