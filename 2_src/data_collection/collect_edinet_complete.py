import os
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# --- 設定: 警告の非表示 ---
warnings.filterwarnings("ignore", category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# --- 初期設定 ---
load_dotenv()
BASE_DRIVE_DIR = Path("C:/M2_Research_Project/1_data/")
EDINET_API_KEY = os.getenv("EDINET_API_KEY")

if not EDINET_API_KEY:
    print("【警告】EDINET_API_KEYが設定されていません。")


class Config:
    # 保存先
    SAVE_FOLDER = BASE_DRIVE_DIR / "edinet_reports" / "01_zip_files_indices"
    META_SAVE_FOLDER = BASE_DRIVE_DIR / "edinet_reports" / "00_metadata"
    META_FILE_NAME = "metadata_2018_2025_all.csv"

    # ターゲット銘柄リストの場所
    INDICES_DIR = BASE_DRIVE_DIR / "raw"
    INDEX_FILES = [
        "growth_250.csv",
        "growth_core.csv",
        "jpx_nikkei_400.csv",
        "nikkei_225.csv",
        "topix_100.csv",
        "topix_core30.csv",
    ]

    # API設定
    API_ENDPOINT_LIST = "https://disclosure.edinet-fsa.go.jp/api/v2/documents.json"
    API_ENDPOINT_DOC = "https://disclosure.edinet-fsa.go.jp/api/v2/documents"

    # 収集期間
    START_DATE = date(2018, 1, 1)
    END_DATE = date.today()

    # 収集対象（有報、四半期、半期、およびその訂正）
    TARGET_DOC_TYPES = ["120", "130", "140", "150", "160", "170"]

    # APIリクエスト間隔
    SLEEP_TIME = 0.5


def make_dirs():
    Config.SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
    Config.META_SAVE_FOLDER.mkdir(parents=True, exist_ok=True)


def load_target_universe():
    """6つのCSVファイルを読み込み、ターゲット証券コードのセットを作成"""
    print("\n--- ターゲット銘柄リストの作成 ---")
    target_codes = set()

    for filename in Config.INDEX_FILES:
        file_path = Config.INDICES_DIR / filename

        # ファイルが見つからない場合のフォールバック
        if not file_path.exists():
            # 直下を探す
            if Path(filename).exists():
                file_path = Path(filename)
            else:
                print(f"警告: ファイルが見つかりません {filename}")
                continue

        try:
            df = pd.read_csv(file_path)
            # 'code' カラムを文字列にし、4桁で統一する (例: 72030 -> 7203)
            if "code" in df.columns:
                codes = df["code"].astype(str).str[:4].tolist()
                target_codes.update(codes)
                print(f"  Loaded {len(codes)} codes from {filename}")
            else:
                print(f"  Error: 'code' column not found in {filename}")
        except Exception as e:
            print(f"  Error reading {filename}: {e}")

    print(f"統合されたターゲット銘柄数 (重複除外): {len(target_codes)} 社")
    return target_codes


def get_document_list(session, target_date):
    """指定日の書類一覧を取得（Session使用）"""
    params = {"date": target_date.strftime("%Y-%m-%d"), "type": 2, "Subscription-Key": EDINET_API_KEY}
    try:
        res = session.get(Config.API_ENDPOINT_LIST, params=params, verify=False, timeout=30)
        if res.status_code == 200:
            return res.json().get("results", [])
        elif res.status_code == 404:
            return []
        else:
            print(f"\nError {res.status_code} on {target_date}")
            return []
    except Exception as e:
        print(f"\nException on {target_date}: {e}")
        return []


def collect_metadata_robust():
    """メタデータを収集する（途中再開対応版）"""
    print("\n--- Step 1: メタデータ収集 ---")
    csv_path = Config.META_SAVE_FOLDER / Config.META_FILE_NAME

    # 1. 再開位置の決定
    if csv_path.exists():
        try:
            # DtypeWarning回避のためlow_memory=False
            df_existing = pd.read_csv(csv_path, low_memory=False)
            if not df_existing.empty and "submitDateTime" in df_existing.columns:
                last_dt_str = df_existing["submitDateTime"].max()
                if pd.notna(last_dt_str):
                    last_date = pd.to_datetime(last_dt_str).date()
                    start_date = last_date + timedelta(days=1)
                    print(f"既存データあり: {last_date} まで取得済み。再開: {start_date}")
                else:
                    start_date = Config.START_DATE
            else:
                start_date = Config.START_DATE
        except Exception as e:
            print(f"既存メタデータの読み込みエラー: {e}。新規作成します。")
            start_date = Config.START_DATE
    else:
        print("新規作成します。")
        start_date = Config.START_DATE
        pd.DataFrame(
            columns=[
                "docID",
                "docTypeCode",
                "secCode",
                "filerName",
                "submitDateTime",
                "docDescription",
                "periodStart",
                "periodEnd",
            ]
        ).to_csv(csv_path, index=False, encoding="utf-8-sig")

    if start_date > Config.END_DATE:
        print("全ての期間のメタデータ収集が完了しています。")
        # 読み込んで返す
        return pd.read_csv(csv_path, dtype=str)

    # 2. ループ処理
    current_date = start_date
    days_to_process = (Config.END_DATE - current_date).days + 1
    buffer = []

    with requests.Session() as session:
        with tqdm(total=days_to_process, desc="Metadata Fetching") as pbar:
            while current_date <= Config.END_DATE:
                docs = get_document_list(session, current_date)

                if docs:
                    for doc in docs:
                        buffer.append(
                            {
                                "docID": doc.get("docID"),
                                "docTypeCode": doc.get("docTypeCode"),
                                "secCode": doc.get("secCode"),
                                "filerName": doc.get("filerName"),
                                "submitDateTime": doc.get("submitDateTime"),
                                "docDescription": doc.get("docDescription"),
                                "periodStart": doc.get("periodStart"),
                                "periodEnd": doc.get("periodEnd"),
                            }
                        )

                # 定期保存
                if len(buffer) > 0 and (current_date.day == 1 or current_date == Config.END_DATE):
                    df_chunk = pd.DataFrame(buffer)
                    df_chunk.to_csv(csv_path, mode="a", header=False, index=False, encoding="utf-8-sig")
                    buffer = []

                current_date += timedelta(days=1)
                pbar.update(1)
                time.sleep(0.1)

    print(f"メタデータ収集完了: {csv_path}")
    # 重複排除して保存し直す（その際、型を文字列として読み込む）
    df_final = pd.read_csv(csv_path, dtype=str)
    df_final = df_final.drop_duplicates(subset=["docID"])
    df_final.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return df_final


def download_filtered_reports(df_meta, target_universe):
    """
    Step 2: ターゲット銘柄の書類のみダウンロード
    """
    print("\n--- Step 2: ターゲット銘柄の書類ダウンロード ---")

    # 1. データ型の補正 (float -> str, ".0"削除)
    # docTypeCodeを文字列にし、'120.0' -> '120' に直す
    df_meta["docTypeCode"] = df_meta["docTypeCode"].astype(str).str.replace(".0", "", regex=False)

    # secCodeも同様
    df_meta["secCode"] = df_meta["secCode"].astype(str).str.replace(".0", "", regex=False)
    # 5桁の場合は先頭4桁を取る ('72030' -> '7203')
    df_meta["code_4digit"] = df_meta["secCode"].str[:4]

    # 2. 書類種別でフィルタ
    # 文字列同士で比較できるようになったので安全
    df_filtered = df_meta[df_meta["docTypeCode"].isin(Config.TARGET_DOC_TYPES)].copy()

    # 3. ターゲット銘柄リストに含まれるかチェック
    # target_universe は文字列のセット {'7203', '6758', ...}
    df_target = df_filtered[df_filtered["code_4digit"].isin(target_universe)].copy()

    print(f"メタデータ全件数: {len(df_meta)}")
    print(f"種別フィルタ後 (有報・四半期等): {len(df_filtered)}")
    print(f"銘柄フィルタ後 (ダウンロード対象): {len(df_target)}")

    if df_target.empty:
        print("ダウンロード対象がありません。銘柄リストや期間を確認してください。")
        # デバッグ用: 先頭のデータを見せる
        print("--- Debug Info ---")
        print("Target Doc Types:", Config.TARGET_DOC_TYPES)
        print("Sample Metadata DocType:", df_meta["docTypeCode"].unique()[:10])
        print("Sample Target Universe:", list(target_universe)[:10])
        return

    # 4. ダウンロード実行
    success_count = 0
    skip_count = 0
    error_count = 0

    with requests.Session() as session:
        for index, row in tqdm(df_target.iterrows(), total=len(df_target), desc="Downloading"):
            doc_id = row["docID"]
            save_path = Config.SAVE_FOLDER / f"{doc_id}.zip"

            # 既存チェック
            if save_path.exists() and save_path.stat().st_size > 0:
                skip_count += 1
                continue

            url = f"{Config.API_ENDPOINT_DOC}/{doc_id}"
            params = {"type": 1, "Subscription-Key": EDINET_API_KEY}

            try:
                res = session.get(url, params=params, stream=True, verify=False, timeout=60)

                if res.status_code == 200:
                    with open(save_path, "wb") as f:
                        for chunk in res.iter_content(chunk_size=8192):
                            f.write(chunk)
                    success_count += 1
                elif res.status_code == 404:
                    error_count += 1
                else:
                    error_count += 1

                time.sleep(Config.SLEEP_TIME)

            except Exception as e:
                print(f"Error {doc_id}: {e}")
                error_count += 1
                time.sleep(1)

    print("\n--- ダウンロード完了 ---")
    print(f"成功(新規): {success_count}")
    print(f"スキップ(既存): {skip_count}")
    print(f"エラー: {error_count}")


def main():
    make_dirs()

    # 1. ターゲット銘柄の読み込み
    target_universe = load_target_universe()
    if not target_universe:
        print("ターゲット銘柄が読み込めませんでした。処理を中止します。")
        return

    # 2. メタデータ収集
    df_meta = collect_metadata_robust()

    # 3. フィルタリングしてダウンロード
    if not df_meta.empty:
        download_filtered_reports(df_meta, target_universe)
    else:
        print("メタデータがありません。")


if __name__ == "__main__":
    main()
