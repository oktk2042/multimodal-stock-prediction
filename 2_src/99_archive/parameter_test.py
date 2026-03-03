import os
import requests
from dotenv import load_dotenv

# --- 設定 ---
load_dotenv()
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
if not NEWS_API_KEY:
    raise ValueError("環境変数 'NEWS_API_KEY' が設定されていません。")

# テストするパラメータ
SEARCH_IN = 'title,description'
DOMAINS_TO_SEARCH = ",".join(['nikkei.com', 'toyokeizai.net', 'itmedia.co.jp', 'diamond.jp'])

# --- テスト実行 ---
print("どのパラメータが問題かを切り分けるテストを開始します...")

try:
    # どのパラメータが問題か切り分けるため、1つずつ有効にして試します
    params = {
        'q': 'トヨタ',
        'language': 'jp',
        'pageSize': 5,
        
        # --- ここを編集してテストします ---
        # ステップ2でこの行のコメントを外す
        # 'domains': DOMAINS_TO_SEARCH, 
        
        # ステップ3でこの行のコメントを外す
        # 'searchIn': SEARCH_IN,
    }

    print("News APIにリクエストを送信中...")
    print("使用するパラメータ:", params)
    
    response = requests.get(
        "https://newsapi.org/v2/everything",
        headers={'X-Api-Key': NEWS_API_KEY},
        params=params,
        timeout=20
    )
    
    response.raise_for_status()
    data = response.json()

    print(f"✅ 成功しました！ 記事を {len(data.get('articles', []))}件取得しました。")

except Exception as e:
    print(f"❌ 失敗しました。以下のエラーが発生しました:")
    print(e)