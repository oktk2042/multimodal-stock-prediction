import os
import requests
import pandas as pd
from dotenv import load_dotenv

def test_api_key():
    """
    .envファイルからAPIキーを読み込み、簡単なテストクエリを実行する。
    """
    # 1. .envファイルからAPIキーを読み込む
    load_dotenv()
    api_key = os.getenv('NEWS_API_KEY')

    if not api_key:
        print("エラー: .envファイルに'NEWS_API_KEY'が見つかりません。")
        return

    # 2. APIリクエストの準備
    url = 'https://newsapi.org/v2/everything'
    params = {
        'q': 'Toyota',  # 確実に結果があるはずのキーワードでテスト
        'language': 'jp',
        'sortBy': 'publishedAt',
        'pageSize': 10  # テストなので件数は少なく設定
    }
    headers = {'X-Api-Key': api_key}

    print("APIキーのテストを開始します...")

    # 3. APIにリクエストを送信
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()  # エラーがあれば例外を発生させる

        data = response.json()
        total_results = data.get('totalResults', 0)
        articles = data.get('articles', [])

        print(f"APIからのレスポンス: 成功 (ステータスコード {response.status_code})")
        print(f"「トヨタ」に関する記事の総数: {total_results}件")

        # 4. 結果をDataFrameで表示
        if articles:
            print("\n取得した記事のサンプル:")
            df = pd.DataFrame(articles)
            # 表示する横幅を調整
            pd.options.display.max_colwidth = 50
            print(df[['publishedAt', 'title', 'url']])
            print("\n✅ テスト成功: APIキーは正常に機能しています。")
        else:
            print("\n❌ テスト失敗: 記事が0件でした。APIキーまたはアカウントの状態を確認してください。")

    except requests.exceptions.RequestException as e:
        print(f"\n❌ テスト失敗: APIリクエスト中にエラーが発生しました。")
        print(f"詳細: {e}")


if __name__ == "__main__":
    test_api_key()