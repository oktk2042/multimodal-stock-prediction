import os
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- 初期設定 ---
load_dotenv()
JQUANTS_REFRESH_TOKEN = os.getenv('JQUANTS_REFRESH_TOKEN')
if not JQUANTS_REFRESH_TOKEN:
    raise ValueError(".envファイルに 'JQUANTS_REFRESH_TOKEN' を設定してください。")

# --- ファイルパス等の設定 ---
BASE_DATA_DIR = "1_data"
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
SEARCH_MAP_PATH = os.path.join(PROCESSED_DIR, "company_search_map.json") 
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "tdnet_disclosures.csv")
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
    jquants_code = f"{code}0" # J-Quantsの銘柄コード形式
    url = f"{API_URL}/fins/timely_disclosure"
    headers = {'Authorization': f'Bearer {id_token}'}
    params = {'code': jquants_code, 'from': start_date, 'to': end_date}
    
    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        return response.json().get('disclosure', [])
    except requests.exceptions.RequestException as e:
        # 404エラーはデータがないだけなので、静かに処理
        if hasattr(e, 'response') and e.response.status_code == 404:
            return []
        tqdm.write(f"  -> [{code}] APIエラー: {e}")
        return []

def collect_all_disclosures():
    """全銘柄の適時開示情報を収集し、CSVに保存する"""
    id_token = get_jquants_id_token(JQUANTS_REFRESH_TOKEN)
    if not id_token:
        return

    try:
        with open(SEARCH_MAP_PATH, 'r', encoding='utf-8') as f:
            search_map = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 検索マップファイルが見つかりません: {SEARCH_MAP_PATH}")
        return
        
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365) # 過去1年分を取得
    
    all_disclosures = []
    
    for code in tqdm(search_map.keys(), desc="TDnet情報収集中"):
        disclosures = fetch_timely_disclosures(
            id_token, 
            code, 
            start_date.strftime('%Y-%m-%d'), 
            end_date.strftime('%Y-%m-%d')
        )
        if disclosures:
            all_disclosures.extend(disclosures)
        
        # J-Quants APIのレートリミットを考慮して待機
        time.sleep(0.5)

    if not all_disclosures:
        print("指定された期間に、対象銘柄の適時開示情報は見つかりませんでした。")
        return

    df = pd.DataFrame(all_disclosures)
    df.sort_values(by="Date", ascending=False, inplace=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n★★★ 完了: {len(df)}件の適時開示情報を {OUTPUT_CSV} に保存しました。★★★")

if __name__ == "__main__":
    collect_all_disclosures()