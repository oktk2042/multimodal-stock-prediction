# --- ライブラリのインポート ---
import os
import time
import json
import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from tqdm import tqdm
import warnings
from dotenv import load_dotenv

# urllib3のInsecureRequestWarningを非表示にする
warnings.filterwarnings('ignore', category=requests.packages.urllib3.exceptions.InsecureRequestWarning)

# --- 初期設定 ---
load_dotenv() # プロジェクトルートの.envファイルを読み込み
BASE_DRIVE_DIR = Path("./1_data/edinet_reports/")
EDINET_API_KEY = os.getenv('EDINET_API_KEY')
if not EDINET_API_KEY:
    raise ValueError("プロジェクトルートの .env ファイルに 'EDINET_API_KEY' を設定してください。")

# --- 設定項目を管理するクラス ---
class Config:
    """設定を管理するクラス"""
    BASE_DIR = BASE_DRIVE_DIR
    # ダウンロードしたZIPファイルの保存先
    SAVE_FOLDER = BASE_DIR / "01_zip_files/"
    # 収集対象の企業リストが書かれたJSONファイルのパス
    SEARCH_MAP_PATH = Path("./1_data/processed/company_search_map.json") # 入力元のパスを設定
    # EDINET APIキー
    API_KEY = EDINET_API_KEY
    # データの信頼性を担保するため、何日分遡ってデータを再取得するか
    RELIABILITY_DAYS = 7
    # 初回実行時に何年分のデータを取得するか
    INITIAL_FETCH_YEARS = 5
    # ダウンロード対象の書類タイプコード
    # 120: 有価証券報告書, 140: 四半期報告書, 160: 半期報告書
    TARGET_DOC_TYPE_CODES = ['120', '140', '160']

# --- 関数定義 ---

def load_target_sec_codes(search_map_path: Path) -> list:
    """
    指定されたJSONファイルから証券コードを読み込み、EDINET API仕様のリストを返す。
    例: '1332' -> '13320'
    """
    print("--- 企業リストの読み込み ---")
    if not search_map_path.exists():
        raise FileNotFoundError(f"企業リストファイルが見つかりません: {search_map_path}")

    with open(search_map_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSONのキー（証券コード）を取得し、末尾に'0'を追加してリスト化
    sec_codes = [f"{code}0" for code in data.keys()]
    
    print(f"  - {len(sec_codes)} 社を収集対象とします。")
    print("-" * 40 + "\n")
    return sec_codes

def update_summary_file(base_dir: Path, api_key: str) -> pd.DataFrame:
    """EDINETから日次の書類一覧を取得し、サマリーファイルを更新する。"""
    summary_path = base_dir / "EDINET_Summary_v3.csv"
    print(f"サマリーファイル '{summary_path.name}' の状態を確認・更新します...")

    today = date.today()
    summary = pd.DataFrame()
    start_day = today - timedelta(days=365 * Config.INITIAL_FETCH_YEARS)

    if summary_path.exists():
        try:
            dtype_map = {'secCode': str, 'docTypeCode': str, 'xbrlFlag': str, 'csvFlag': str}
            summary = pd.read_csv(summary_path, encoding='utf_8_sig', dtype=dtype_map)
            if not summary.empty:
                summary['submitDateTime'] = pd.to_datetime(summary['submitDateTime'], errors='coerce')
                latest_date_in_file = summary['submitDateTime'].max().date()
                start_day = latest_date_in_file - timedelta(days=Config.RELIABILITY_DAYS)
        except Exception as e:
            print(f"サマリーファイルの読み込み中にエラーが発生しました: {e}")

    end_day = today
    day_term = [start_day + timedelta(i) for i in range((end_day - start_day).days + 1)]

    new_docs = []
    # tqdmのnotebook版から標準版に変更
    for day in tqdm(day_term, desc="APIからメタデータ取得"):
        params = {'date': day.strftime('%Y-%m-%d'), 'type': 2, 'Subscription-Key': api_key}
        try:
            response = requests.get('https://disclosure.edinet-fsa.go.jp/api/v2/documents.json', params=params, verify=False, timeout=30)
            response.raise_for_status()
            res_json = response.json()
            if res_json.get('results'):
                new_docs.extend(res_json['results'])
        except requests.exceptions.RequestException as e:
            tqdm.write(f"エラー: {day} のデータ取得に失敗 - {e}")
        time.sleep(0.1)

    if new_docs:
        temp_df = pd.DataFrame(new_docs)
        summary = pd.concat([summary, temp_df], ignore_index=True)

    if not summary.empty:
        summary['submitDateTime'] = pd.to_datetime(summary['submitDateTime'], errors='coerce')
        summary.dropna(subset=['docID'], inplace=True)
        summary = summary.drop_duplicates(subset='docID', keep='last')
        summary = summary.sort_values(by='submitDateTime', ascending=True).reset_index(drop=True)
        
    # 出力先フォルダがなければ作成
    base_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False, encoding='utf_8_sig')
    print("\n✅ サマリーファイルの更新が完了しました！")
    return summary

def step1_create_and_summarize():
    """ステップ①: サマリーを作成し、その概要を出力する。"""
    print("--- ステップ① サマリー作成と概要の表示 ---")
    summary_df = update_summary_file(Config.BASE_DIR, Config.API_KEY)

    if summary_df.empty:
        print("⚠️ サマリーデータの作成に失敗したか、データがありませんでした。")
        return pd.DataFrame()

    print(f"  - データ期間: {summary_df['submitDateTime'].min():%Y-%m-%d} ～ {summary_df['submitDateTime'].max():%Y-%m-%d}")
    print(f"  - 総データ数: {len(summary_df)} 件")
    print(f"  - 銘柄数（ユニーク）: {summary_df['secCode'].nunique()} 社")
    print("-" * 40 + "\n")
    return summary_df

def step2_check_download_status(summary_df: pd.DataFrame, target_sec_codes: list):
    """ステップ②: ダウンロード状況を確認し、未ダウンロードの書類を特定する。"""
    print("--- ステップ② ダウンロード状況の確認 ---")
    save_folder = Config.SAVE_FOLDER
    save_folder.mkdir(parents=True, exist_ok=True)

    existing_files_path = list(save_folder.rglob('*.zip'))
    print(f"📁 指定フォルダ: {save_folder}")
    print(f"  - 既存ファイル数: {len(existing_files_path)} 件")

    # --- 対象企業リストを使って絞り込み ---
    query_str = (
        "csvFlag == '1' and "
        "secCode.notna() and secCode != 'None' and "
        f"docTypeCode in {Config.TARGET_DOC_TYPE_CODES} and "
        "secCode in @target_sec_codes"  # 収集対象の証券コードリストでフィルタ
    )
    target_docs = summary_df.query(query_str)

    existing_file_stems = {f.stem for f in existing_files_path}
    docs_to_download = target_docs[~target_docs['docID'].isin(existing_file_stems)]

    print("\n📊 サマリーと照合した結果:")
    print(f"  - ダウンロード対象の総書類数（指定企業・CSV提供あり）: {len(target_docs)} 件")
    print(f"  - ダウンロードが必要な（未取得の）書類数: {len(docs_to_download)} 件")
    print("-" * 40 + "\n")
    return docs_to_download

def step3_execute_download(docs_to_download: pd.DataFrame):
    """ステップ③: ファイルのダウンロードを実行し、年/四半期フォルダに保存する。"""
    print("--- ステップ③ ダウンロードの実行 ---")
    if docs_to_download.empty:
        print("✅ ダウンロード対象の新しいファイルはありません。処理を完了します。")
        print("-" * 40 + "\n")
        return

    print(f"{len(docs_to_download)}件のファイルのダウンロードを開始します。")
    for _, row in tqdm(docs_to_download.iterrows(), total=len(docs_to_download), desc="ZIPダウンロード進捗"):
        doc_id = row['docID']
        submit_date = row['submitDateTime']

        if pd.isna(submit_date):
            target_folder = Config.SAVE_FOLDER / "unknown_date"
        else:
            year = submit_date.year
            quarter = (submit_date.month - 1) // 3 + 1
            target_folder = Config.SAVE_FOLDER / str(year) / f"Q{quarter}"
        
        target_folder.mkdir(parents=True, exist_ok=True)
        zip_path = target_folder / f"{doc_id}.zip"
        
        url_zip = f"https://disclosure.edinet-fsa.go.jp/api/v2/documents/{doc_id}"
        params_zip = {"type": 5, 'Subscription-Key': Config.API_KEY}

        try:
            r = requests.get(url_zip, params=params_zip, stream=True, verify=False, timeout=60)
            r.raise_for_status()
            with open(zip_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            tqdm.write(f"ダウンロード失敗: {doc_id}, エラー: {e}")
            if zip_path.exists():
                zip_path.unlink()
        time.sleep(0.1)

    print("\n✅ ダウンロード処理が完了しました。")
    print("-" * 40 + "\n")

def main():
    """メイン処理を実行する関数"""
    try:
        # 収集対象の企業リストを読み込み
        target_codes = load_target_sec_codes(Config.SEARCH_MAP_PATH)
        
        # ステップ①: メタデータのサマリーを作成・更新
        summary_data = step1_create_and_summarize()

        if not summary_data.empty:
            # ステップ②: ダウンロードが必要なファイルを特定
            files_to_download = step2_check_download_status(summary_data, target_codes)
            
            # ステップ③: ダウンロードを実行
            step3_execute_download(files_to_download)
        
        print("全ての処理が完了しました。🎉")

    except Exception as e:
        print("\n--- 予期せぬエラーが発生しました ---")
        print(e)
        print("---------------------------------")

# --- スクリプトとして実行 ---
if __name__ == '__main__':
    main()