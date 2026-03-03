import json
import os
import re

import pandas as pd
import requests
from dotenv import load_dotenv

# --- 設定 ---
PROCESSED_DIR = "1_data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)


def get_jquants_id_token(refresh_token: str) -> str:
    print("J-QuantsからIDトークンを取得中...")
    url = f"https://api.jquants.com/v1/token/auth_refresh?refreshtoken={refresh_token}"
    try:
        response = requests.post(url)
        response.raise_for_status()
        id_token = response.json().get("idToken")
        print("-> IDトークンの取得完了。")
        return id_token
    except requests.exceptions.RequestException as e:
        print(f"エラー: IDトークンの取得に失敗しました。詳細: {e}")
        return None


def get_listed_info_data(id_token: str) -> dict:
    print("J-Quantsから全銘柄のリストを取得中...")
    url = "https://api.jquants.com/v1/listed/info"
    headers = {"Authorization": f"Bearer {id_token}"}
    try:
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        listed_info = response.json().get("info", [])
        info_map = {item["Code"]: item for item in listed_info}
        print(f"-> {len(info_map)}件の銘柄情報を取得完了。")
        return info_map
    except requests.exceptions.RequestException as e:
        print(f"エラー: 上場銘柄情報の取得に失敗しました。詳細: {e}")
        return {}


def generate_company_map_from_master_list():
    """master_stock_list.csvを基に、EDINETコードを含むリッチな検索マップを生成する"""
    load_dotenv()
    refresh_token = os.getenv("JQUANTS_REFRESH_TOKEN")
    if not refresh_token:
        print("エラー: .envファイルに'JQUANTS_REFRESH_TOKEN'が見つかりません。")
        return

    id_token = get_jquants_id_token(refresh_token)
    if not id_token:
        return
    listed_info_data = get_listed_info_data(id_token)
    if not listed_info_data:
        return
    master_list_path = os.path.join(PROCESSED_DIR, "master_stock_list.csv")
    try:
        master_df = pd.read_csv(master_list_path, dtype={"code": str})
        print(f"\n{master_list_path} から {len(master_df)}件の銘柄を読み込みました。")
    except FileNotFoundError:
        print(f"エラー: {master_list_path} が見つかりません。")
        return

    company_search_map = {}
    print("リッチ検索キーワードマップを生成中...")

    for index, row in master_df.iterrows():
        code = row["code"]
        common_name = row["name"]
        jquants_code = f"{code}0"
        info = listed_info_data.get(jquants_code)

        if info:
            # ▼▼▼ ここからが修正・追加箇所 ▼▼▼
            official_name = info.get("CompanyName", "")
            english_name = info.get("CompanyNameEnglish", "")
            sector = info.get("Sector33CodeName", "")
            edinet_code = info.get("EDINETCode", "")  # EDINETコードを取得

            keywords = {official_name, english_name, common_name}
            if english_name:
                simple_english = re.sub(r",? (Inc|Ltd|Co|Corporation)\.?$", "", english_name).strip()
                keywords.add(simple_english)

            keywords = sorted([kw for kw in keywords if kw])

            company_search_map[code] = {
                "search_keywords": keywords,
                "sector": sector,
                "edinet_code": edinet_code,  # 取得したEDINETコードをマップに追加
            }
            # ▲▲▲ ここまでが修正・追加箇所 ▲▲▲
        else:
            print(f"警告: 銘柄コード '{code}' ({common_name}) の情報がJ-Quantsに見つかりませんでした。")

    # JSON形式で保存
    json_path = os.path.join(PROCESSED_DIR, "company_search_map.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(company_search_map, f, ensure_ascii=False, indent=4)
    print(f"\n-> リッチ検索マップ(JSON)を {json_path} に保存しました。")


if __name__ == "__main__":
    generate_company_map_from_master_list()
