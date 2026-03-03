import os
import json
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta

# --- 初期設定 ---
load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    raise ValueError("環境変数 'NEWS_API_KEY' が設定されていません。.envファイルを確認してください。")

# --- ファイルパス等の設定 ---
BASE_DATA_DIR = "1_data"
RAW_DIR = os.path.join(BASE_DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
NEWS_SAVE_DIR = os.path.join(RAW_DIR, "news")
SEARCH_MAP_PATH = os.path.join(PROCESSED_DIR, "company_search_map.json")
os.makedirs(NEWS_SAVE_DIR, exist_ok=True)

# 処理対象のインデックスファイル（実行したい行のコメントを外してください）
INDEX_FILES = {
    "topix_core30_tickers.txt": "TOPIX Core30",
    # "topix_100_tickers.txt": "TOPIX 100",
    # "nikkei_225_tickers.txt": "Nikkei 225",
    # "jpx_nikkei_400_tickers.txt": "Jpx Nikkei 400",
    # "growth_core_tickers.txt": "Growth Core",
    # "growth_250_tickers.txt": "Growth 250",
}

# --- News API 基本設定 ---
BASE_URL = 'https://newsapi.org/v2/everything'
HEADERS = {'X-Api-Key': NEWS_API_KEY}
IS_FREE_PLAN = True             # 無料プランの場合はTrue, 有料プランの場合はFalse
MAX_RETRIES = 5                 # エラー時に再試行する最大回数
RETRY_WAIT_SECONDS = 60         # レートリミット時に待機する秒数

BASE_PARAMS = {
    'language': 'jp',
    'pageSize': 100,
    'sortBy': 'publishedAt',                                                            # 並び順 ('publishedAt': 最新順, 'relevancy': 関連順, 'popularity': 人気順)
    'searchIn': 'title,description',                                                    # 検索対象をタイトル(title), 説明文(description)に限定 (より厳密にするなら 'title' のみ)
    'domains': ",".join(['nikkei.com', 'toyokeizai.net', 'itmedia.co.jp', 'diamond.jp', 
                         'bloomberg.co.jp', 'reuters.com', 'businessinsider.jp', 
                         'sbbit.jp', 'impress.co.jp', 'ascii.jp'])                      # 検索対象とするドメインを指定 (信頼できるニュースソースに絞り込む)
}

def fetch_all_articles_for_query(query):
    """指定されたクエリでNews APIにリクエストを送り、全ページの記事を取得する。レートリミットに対応。"""
    for attempt in range(MAX_RETRIES):
        all_articles = []
        page = 1
        total_results = float('inf')

        while len(all_articles) < total_results:
            params = {
                'q': query,
                'language': 'jp',
                'from': START_DATE,
                'to': END_DATE,
                'pageSize': 100,
                'page': page,
                'sortBy': SORT_BY
                # 'searchIn': SEARCH_IN,
                # 'domains': DOMAINS_TO_SEARCH, 
            }
        print(f"    - ページ {page} を取得中 (試行 {attempt + 1}/{MAX_RETRIES})...")
        
        try:
            response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20)
            response.raise_for_status() # 4xx or 5xx エラーの場合に例外を発生
            
            # リクエスト成功
            data = response.json()
            articles_page = data.get('articles', [])
            
            if not articles_page:
                return all_articles # これ以上記事がなければ、現在までの結果を返す
            
            all_articles.extend(articles_page)
            
            # 無料プランの場合は1ページで終了
            if IS_FREE_PLAN:
                return all_articles
                
            # 有料プランの場合のページネーション (現在は未使用)
            total_results = data.get('totalResults', 0)
            if len(all_articles) >= total_results:
                return all_articles
            
            page += 1
            time.sleep(1.5)

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"    - 警告: レートリミットに達しました。{RETRY_WAIT_SECONDS}秒待機します...")
                time.sleep(RETRY_WAIT_SECONDS)
                # ループの次の試行(リトライ)に進む
                continue
            else:
                # 429以外のHTTPエラー
                print(f"    - エラー: HTTPエラーが発生しました: {e}")
                return None # リトライせずに失敗を返す
        
        except requests.exceptions.RequestException as e:
            # タイムアウトなどの接続エラー
            print(f"    - エラー: APIリクエスト中に問題が発生しました: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"    - {RETRY_WAIT_SECONDS}秒待機して再試行します...")
                time.sleep(RETRY_WAIT_SECONDS)
                continue
            else:
                return None # リトライの上限に達したら失敗を返す

    # リトライの上限に達しても成功しなかった場合
    print("    - 失敗: リトライの上限に達しました。")
    return None

def fetch_articles(query: str, start_date: str, end_date: str) -> list | None:
    """
    指定されたクエリでNews APIにリクエストを送る（リトライ機能付き）。
    成功した場合は記事のリスト、失敗した場合はNoneを返す。
    """
    for attempt in range(MAX_RETRIES):
        params = BASE_PARAMS.copy()
        params.update({
            'q': query,
            'from': start_date,
            'to': end_date,
            'page': 1  # 無料プランでは1ページ目のみ
        })
        
        print(f"    - APIにリクエスト送信中 (試行 {attempt + 1}/{MAX_RETRIES})...")
        try:
            response = requests.get(BASE_URL, headers=HEADERS, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            return data.get('articles', []) # 成功したら記事リストを返す

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                print(f"    - 警告: レートリミットです。{RETRY_WAIT_SECONDS}秒待機します...")
            else:
                print(f"    - エラー: HTTPエラーが発生しました: {e}")
        except requests.exceptions.RequestException as e:
            print(f"    - エラー: 接続エラーが発生しました: {e}")
        
        # エラーが発生した場合、最後の試行でなければ待機
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_WAIT_SECONDS)
        
    print("    - 失敗: リトライの上限に達しました。")
    return None

def collect_news_for_all_indices():
    """ティッカーリストと検索マップを使い、各銘柄のニュースを収集する"""
    try:
        with open(SEARCH_MAP_PATH, 'r', encoding='utf-8') as f:
            search_map = json.load(f)
        print("--- 検索キーワードマップを読み込みました ---")
    except FileNotFoundError:
        print(f"エラー: 検索マップファイルが見つかりません: {SEARCH_MAP_PATH}")
        return

    to_date = datetime.now().date()
    end_date_str = to_date.strftime('%Y-%m-%d')
    start_date_str = (to_date - timedelta(days=29)).strftime('%Y-%m-%d')

    for txt_filename, index_name in INDEX_FILES.items():
        ticker_file_path = os.path.join(RAW_DIR, txt_filename)
        index_save_dir = os.path.join(NEWS_SAVE_DIR, index_name.replace(' ', '_').lower())
        os.makedirs(index_save_dir, exist_ok=True)
        print(f"\n--- {index_name} のニュース収集を開始 ---")
        
        with open(ticker_file_path, 'r', encoding='utf-8') as f:
            tickers = [line.strip().replace('.T', '') for line in f if line.strip()]

        for i, code in enumerate(tickers):
            company_data = search_map.get(code)
            if not company_data:
                print(f"  -> スキップ: {code} の検索キーワードがマップに見つかりません。")
                continue
            
            keywords = company_data.get("search_keywords", [])
            stock_name = keywords[0] if keywords else code
            safe_stock_name = stock_name.replace('/', '／')
            output_filename = os.path.join(index_save_dir, f"{code}_{safe_stock_name}_news.csv")

            if os.path.exists(output_filename):
                print(f"  -> 処理済み: {stock_name} ({code})")
                continue
            
            print(f"  -> 処理中: {stock_name} ({code})")
            query = " OR ".join(f'"{kw}"' for kw in keywords)

            # 最初の銘柄以外は、APIリクエストの前に必ず待機する
            if i > 0:
                print(f"    -> 次のリクエストまで12秒待機します...")
                time.sleep(12)

            articles = fetch_articles(query, start_date_str, end_date_str)

            if articles is not None and articles:
                df = pd.DataFrame(articles)
                df['code'] = code
                df['source'] = df['source'].apply(lambda s: s['name'] if isinstance(s, dict) else s)
                df.to_csv(output_filename, index=False, encoding='utf-8-sig')
                print(f"    -> 完了: {len(df)}件の記事を保存しました。")
            elif articles is None:
                print(f"    -> 失敗: APIエラーのため、処理をスキップしました。")
            else:
                print(f"    -> 完了: 関連ニュースは見つかりませんでした。")

if __name__ == "__main__":
    collect_news_for_all_indices()
    print("\n★★★ 全てのニュース収集処理が完了しました ★★★")