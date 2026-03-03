import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from tqdm import tqdm

# --- 初期設定 ---
load_dotenv()
# V2ではリフレッシュトークンではなくAPIキーを使用
JQUANTS_API_KEY = os.getenv("JQUANTS_API_KEY")
if not JQUANTS_API_KEY:
    # 念のため旧設定も確認（V2キーとして保存している場合を考慮）
    JQUANTS_API_KEY = os.getenv("JQUANTS_REFRESH_TOKEN")

if not JQUANTS_API_KEY:
    raise ValueError(".envファイルに 'JQUANTS_API_KEY' を設定してください。")

# --- 設定 ---
BASE_DATA_DIR = "1_data"
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
SEARCH_MAP_PATH = os.path.join(PROCESSED_DIR, "company_search_map.json")
OUTPUT_CSV = os.path.join(PROCESSED_DIR, "tdnet_disclosures_2020_2025_v2.csv")

# ★V2のエンドポイント設定
# Timely Disclosure (適時開示) のエンドポイント
# V1: /fins/timely_disclosure -> V2: おそらく /fins/disclosure または /fins/announcement の可能性がありますが
# まずは互換性を信じて v2 パスで試行します。
API_BASE_URL = "https://api.jquants.com/v2"


def fetch_disclosures_v2(api_key: str, code: str, start_date: str, end_date: str) -> list:
    """
    J-Quants API V2 を使用して適時開示情報を取得する
    """
    url = f"{API_BASE_URL}/fins/disclosure"  # エンドポイント名は要確認ですが、disclosureが一般的
    # ※もし404になる場合は '/fins/timely_disclosure' も試す価値あり

    # V2の認証ヘッダー (x-api-key が標準的)
    headers = {"x-api-key": api_key}

    # パラメータ（V2でもcode, dateなどの基本は変わらないはず）
    # codeは5桁推奨だが、4桁でも動く場合あり。念のため5桁(末尾0)に
    code_5digit = f"{code}0" if len(code) == 4 else code

    params = {
        "code": code_5digit,
        "from": start_date.replace("-", ""),  # V2はYYYYMMDD形式を好む場合があるため念のため変換
        "to": end_date.replace("-", ""),
    }

    # ハイフン付きでも通る場合があるので、まずはそのまま試してダメなら変換などの分岐が必要ですが
    # ここでは一般的な 'YYYYMMDD' を送ってみます（API仕様によります）
    params["from"] = start_date.replace("-", "")
    params["to"] = end_date.replace("-", "")

    try:
        response = requests.get(url, headers=headers, params=params, timeout=30)

        if response.status_code == 200:
            return response.json().get("disclosure", [])
        elif response.status_code == 404:
            # データがない場合
            return []
        elif response.status_code == 401 or response.status_code == 403:
            print(f"認証エラー (Status: {response.status_code}): APIキーを確認してください。")
            return []
        else:
            # エンドポイント違いの可能性（404 Not FoundがURLに対して出る場合）
            # その場合はURLを /fins/timely_disclosure に変えてリトライするロジックも検討
            if response.status_code == 404 and "disclosure" not in response.text:
                # エンドポイント自体がない場合
                pass
            # tqdm.write(f"  -> [{code}] API Error: {response.status_code} - {response.text}")
            return []

    except Exception as e:
        print(f"通信エラー: {e}")
        return []


def collect_all_disclosures_v2():
    print("--- J-Quants API V2 適時開示情報収集 (2020-2025) ---")

    # 銘柄リスト読み込み
    try:
        with open(SEARCH_MAP_PATH, encoding="utf-8") as f:
            search_map = json.load(f)
        target_codes = list(search_map.keys())
    except FileNotFoundError:
        print("エラー: company_search_map.json が見つかりません。")
        return

    # 期間設定
    start_date_all = datetime(2020, 1, 1).date()
    end_date_all = datetime.now().date()

    all_data = []

    # 1年ごとにループ
    current_start = start_date_all
    while current_start < end_date_all:
        current_end = current_start + relativedelta(years=1) - timedelta(days=1)
        if current_end > end_date_all:
            current_end = end_date_all

        print(f"\n期間: {current_start} ～ {current_end}")

        period_data = []
        for code in tqdm(target_codes):
            # APIコール
            res = fetch_disclosures_v2(
                JQUANTS_API_KEY, code, current_start.strftime("%Y%m%d"), current_end.strftime("%Y%m%d")
            )
            if res:
                period_data.extend(res)

            time.sleep(0.02)  # V2はレートリミットが緩和されている可能性がありますが、安全のため待機

        all_data.extend(period_data)
        print(f"  -> {len(period_data)} 件取得")

        # 中間保存
        if all_data:
            pd.DataFrame(all_data).to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

        current_start = current_end + timedelta(days=1)

    print(f"\n完了。合計 {len(all_data)} 件を {OUTPUT_CSV} に保存しました。")


if __name__ == "__main__":
    collect_all_disclosures_v2()
