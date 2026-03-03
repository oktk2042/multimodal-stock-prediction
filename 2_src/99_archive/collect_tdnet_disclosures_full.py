import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

# --- 初期設定 ---
load_dotenv()
JQUANTS_REFRESH_TOKEN = os.getenv('JQUANTS_REFRESH_TOKEN')
if not JQUANTS_REFRESH_TOKEN:
    raise ValueError(".envファイルに 'JQUANTS_REFRESH_TOKEN' を設定してください。")

# --- ファイルパス等の設定 ---
BASE_DATA_DIR = "1_data"
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
# 収集対象の銘柄リスト
SEARCH_MAP_PATH = os.path.join(PROCESSED_DIR, "company_search_map.json") 
# 保存先ファイル名
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "tdnet_disclosures_2020_2025.csv")
API_URL = "https://api.jquants.com/v1"

def get_jquants_id_token(refresh_token: str) -> str:
    """リフレッシュトークンを使ってIDトークンを取得する"""
    url = f"{API_URL}/token/auth_refresh?refreshtoken={refresh_token}"
    try:
        response = requests.post(url)
        response.raise_for_status()
        return response.json().get('idToken')
    except requests.exceptions.RequestException as e:
        print(f"エラー: IDトークンの取得に失敗しました。詳細: {e}")
        return None

def fetch_timely_disclosures(id_token: str, code: str, start_date: str, end_date: str) -> list:
    """指定された銘柄コードと期間の適時開示情報を取得する"""
    jquants_code = f"{code}0" # J-Quantsの銘柄コード形式 (末尾に0をつける)
    url = f"{API_URL}/fins/timely_disclosure"
    headers = {'Authorization': f'Bearer {id_token}'}
    params = {'code': jquants_code, 'from': start_date, 'to': end_date}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get('disclosure', [])
    except requests.exceptions.RequestException as e:
        # 404エラーは「データなし」の意味なので無視してOK
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            return []
        # その他のエラーはログに出す
        # tqdm.write(f"  -> [{code}] APIエラー: {e}")
        return []

def collect_all_disclosures_full_history():
    """全銘柄の2020年からの適時開示情報を収集し、CSVに保存する"""
    print("--- J-Quants 適時開示情報収集 (2020-2025) を開始します ---")
    
    id_token = get_jquants_id_token(JQUANTS_REFRESH_TOKEN)
    if not id_token:
        return

    # 銘柄リストの読み込み
    try:
        with open(SEARCH_MAP_PATH, 'r', encoding='utf-8') as f:
            search_map = json.load(f)
        target_codes = list(search_map.keys())
        print(f"対象銘柄数: {len(target_codes)} 件")
    except FileNotFoundError:
        print(f"エラー: 検索マップファイルが見つかりません: {SEARCH_MAP_PATH}")
        return
        
    # --- 期間設定: 研究期間に合わせて2020年1月から設定 ---
    # 中間発表資料 [cite: 263] に合わせ、2020/01/01 から現在までを取得
    start_date_all = datetime(2020, 1, 1).date()
    end_date_all = datetime.now().date()
    
    all_disclosures = []
    
    # 既存データがあれば読み込んで、重複取得を防ぐ（オプション）
    if os.path.exists(OUTPUT_CSV):
        print("既存のCSVファイルが見つかりました。データを追加します。")
        existing_df = pd.read_csv(OUTPUT_CSV)
        # 既に取得済みのIDなどを管理するロジックをここに入れることも可能
        # 今回はシンプルに全期間取得して上書き/結合する方針とします
    
    # J-Quantsは長期間を一括で取ると重い/エラーになることがあるため、1年ごとに区切って取得
    current_start = start_date_all
    
    while current_start < end_date_all:
        current_end = current_start + relativedelta(years=1) - timedelta(days=1)
        if current_end > end_date_all:
            current_end = end_date_all
            
        period_str = f"{current_start} ～ {current_end}"
        print(f"\n[{period_str}] のデータを収集中...")
        
        period_disclosures = []
        
        # プログレスバーを表示して各銘柄をループ
        for code in tqdm(target_codes, desc=f"Progress ({current_start.year})"):
            disclosures = fetch_timely_disclosures(
                id_token, 
                code, 
                current_start.strftime('%Y-%m-%d'), 
                current_end.strftime('%Y-%m-%d')
            )
            if disclosures:
                period_disclosures.extend(disclosures)
            
            # レートリミット回避のための待機 (重要)
            time.sleep(0.05) 
        
        all_disclosures.extend(period_disclosures)
        print(f"  -> {len(period_disclosures)} 件取得しました。")
        
        # 1年終わるごとに中間保存（万が一止まっても大丈夫なように）
        if all_disclosures:
            df_temp = pd.DataFrame(all_disclosures)
            df_temp.drop_duplicates(subset=['DocumentID'], inplace=True) # 重複排除
            df_temp.sort_values(by="Date", ascending=False, inplace=True)
            df_temp.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
            print(f"  -> 中間保存完了: {OUTPUT_CSV}")

        # 次の期間へ
        current_start = current_end + timedelta(days=1)

    print("\n★★★ 全処理完了 ★★★")
    print(f"最終的に {len(all_disclosures)} 件の適時開示情報を保存しました。")

if __name__ == "__main__":
    collect_all_disclosures_full_history()