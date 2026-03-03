import os
import io
import json
import time
import zipfile
import requests
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timedelta
from dotenv import load_dotenv
from edinet_xbrl.edinet_xbrl_parser import EdinetXbrlParser

# --- 初期設定 ---
load_dotenv()
EDINET_API_KEY = os.getenv('EDINET_API_KEY')
if not EDINET_API_KEY:
    raise ValueError("環境変数 'EDINET_API_KEY' が設定されていません。")

# --- ファイルパス等の設定 ---
BASE_DATA_DIR = "1_data"
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(BASE_DATA_DIR, "edinet_reports")
SEARCH_MAP_PATH = os.path.join(PROCESSED_DIR, "company_search_map.json") 
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "edinet_fundamental_data.csv")
API_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_documents_list(date_str: str) -> list:
    """指定された日付の書類一覧を取得する"""
    print(f"{date_str} の書類一覧を取得中...")
    url = f"{API_URL}/documents.json"
    
    params = {
        "date": date_str, 
        "type": 2,
        "Subscription-Key": EDINET_API_KEY 
    }
    
    # ヘッダーは空にする
    res = requests.get(url, params=params, headers={})
    res.raise_for_status()
    return res.json().get("results", [])


def download_report(doc_id):
    """指定されたdocIDのXBRLファイルをダウンロードする"""
    url = f"{API_URL}/documents/{doc_id}"
    params = {
        "type": 1, # XBRLはこちらのタイプが適切
        "Subscription-Key": EDINET_API_KEY
    }
    
    res = requests.get(url, params=params, headers={})
    res.raise_for_status()
    folder = os.path.join(OUTPUT_DIR, doc_id)
    if not os.path.exists(folder):
        with zipfile.ZipFile(io.BytesIO(res.content)) as z:
            z.extractall(folder)
    return folder

def parse_xbrl(folder):
    parser = EdinetXbrlParser()
    TAG_MAP = {"発行済株式総数": "NumberOfIssuedSharesTotal", "売上高": "NetSales", "営業利益": "OperatingIncome", "当期純利益": "ProfitLoss", "総資産": "TotalAssets", "自己資本比率": "EquityToAssetRatio", "事業の内容": "DescriptionOfBusiness", "事業等のリスク": "BusinessRisks"}
    data = {}
    xbrl_file_path = None
    for root, _, files in os.walk(folder):
        for f in files:
            if f.endswith(".xbrl") and "PublicDoc" in root:
                xbrl_file_path = os.path.join(root, f)
                break
        if xbrl_file_path: break
    if xbrl_file_path:
        try:
            xbrl = parser.parse(xbrl_file_path)
            for key_jp, tag_name in TAG_MAP.items():
                value_obj = xbrl.get_data_by_tag_name(tag_name)
                data[key_jp] = value_obj.get_value() if value_obj else None
            return data
        except Exception as e:
            print(f"XBRL解析失敗: {e}")
    return data

def collect_edinet_data():
    """メイン処理: EDINETから財務データを収集し、CSVに保存する"""
    
    target_year = 2025 # 対象としたい年を指定
    start_date = datetime(target_year, 4, 1).date()
    end_date = datetime(target_year, 6, 30).date()
    all_docs = []
    date_range = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    print(f"--- {start_date} から {end_date} までの書類を検索します ---")
    for date in tqdm(date_range, desc="書類リスト取得中"):
        date_str = date.strftime('%Y-%m-%d')
        try:
            daily_docs = get_documents_list(date_str)
            if daily_docs:
                all_docs.extend(daily_docs)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code != 404:
                print(f"  -> {date_str} のデータ取得で予期せぬエラー: {e}")
        time.sleep(1.1)
    yuhou_docs = [
        d for d in all_docs 
        if "有価証券報告書" in d.get("docDescription", "") and "訂正" not in d.get("docDescription", "")
    ]
    yuhou_docs.sort(key=lambda x: x.get("submitDateTime"), reverse=True)
    docs_by_edinet_code = {d["edinetCode"]: d for d in reversed(yuhou_docs)}
    print(f"\n対象期間の有価証券報告書件数: {len(docs_by_edinet_code)}")

    try:
        with open(SEARCH_MAP_PATH, 'r', encoding='utf-8') as f:
            search_map = json.load(f)
    except FileNotFoundError:
        print(f"エラー: 検索マップファイルが見つかりません: {SEARCH_MAP_PATH}")
        return

    summary_records = []
    for code, info in tqdm(search_map.items(), desc="銘柄処理中"):
        edinet_code = info.get("edinet_code")
        if not edinet_code or edinet_code not in docs_by_edinet_code:
            continue
        
        target_doc = docs_by_edinet_code.get(edinet_code)
        doc_id = target_doc["docID"]
        
        try:
            folder = download_report(doc_id)
            data = parse_xbrl(folder)
            record = {"Code": code, "EDINETCode": edinet_code, "DocID": doc_id, "SubmitDate": target_doc.get("submitDateTime")}
            record.update(data)
            summary_records.append(record)
        except Exception as e:
            print(f"[{code}] ダウンロード/解析失敗: {e}")
        time.sleep(1.1)

    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n完了: {len(summary_df)}件の企業データを {OUTPUT_CSV} に保存しました。")

def collect_data():
    """【最終テスト用】過去年（2024年）のデータ取得を試みる"""
    
    # 提出が確実にあるはずの「昨年」の平日を1日だけ指定
    test_date = "2024-06-27"
    
    print(f"--- 最終テスト開始：{test_date}の書類取得を試みます ---")

    try:
        # エラーを隠さずに直接APIを呼び出す
        docs = get_documents_list(test_date)
        
        if docs:
            yuhou_docs = [d for d in docs if d.get("formCode") == "030000" and d.get("docTypeCode") == "120"]
            print(f"✅ 成功: {test_date} に {len(docs)}件の書類が見つかりました。")
            print(f"  -> そのうち有価証券報告書は {len(yuhou_docs)}件でした。")
        else:
            print(f"-> 成功しましたが、{test_date} に提出された書類はありませんでした。")

    except requests.exceptions.RequestException as e:
        print(f"❌ 失敗しました。以下のエラーが発生しました:")
        print(e)

if __name__ == "__main__":
    # collect_data()
    collect_edinet_data()