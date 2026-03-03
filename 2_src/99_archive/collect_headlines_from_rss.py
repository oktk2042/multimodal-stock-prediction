import os
import json
import feedparser
import pandas as pd
from datetime import datetime

# --- 設定 ---
BASE_DATA_DIR = "1_data"
PROCESSED_DIR = os.path.join(BASE_DATA_DIR, "processed")
SEARCH_MAP_PATH = os.path.join(PROCESSED_DIR, "company_search_map.json")
OUTPUT_CSV_PATH = os.path.join(PROCESSED_DIR, "collected_headlines.csv")

# 巡回対象のRSSフィードのリスト（自由に追加・編集してください）
RSS_FEEDS = {
    "Reuters Japan (Business)": "https://feeds.reuters.com/reuters/JPBusinessNews",
    "Reuters Japan (Technology)": "https://feeds.reuters.com/reuters/JPTechnologyNews",
    "Yahoo! News (Business)": "https://news.yahoo.co.jp/rss/categories/business.xml",
    "ITmedia (Top Business)": "https://rss.itmedia.co.jp/rss/2.0/news_business.xml",
    "Toyo Keizai Online (All)": "https://toyokeizai.net/list/feed/rss"
}

def collect_headlines():
    """RSSフィードを巡回し、指定した企業のキーワードに一致するヘッドライン情報を収集・追記する。"""
    try:
        with open(SEARCH_MAP_PATH, 'r', encoding='utf-8') as f:
            search_map = json.load(f)
        print(f"--- {os.path.basename(SEARCH_MAP_PATH)} から検索キーワードを読み込みました ---")
    except FileNotFoundError:
        print(f"エラー: 検索マップファイルが見つかりません: {SEARCH_MAP_PATH}")
        return

    # 既存のヘッドラインデータを読み込む
    if os.path.exists(OUTPUT_CSV_PATH):
        existing_df = pd.read_csv(OUTPUT_CSV_PATH)
        seen_urls = set(existing_df['url'].tolist())
        print(f"既存のヘッドライン {len(seen_urls)} 件を読み込みました。")
    else:
        existing_df = pd.DataFrame()
        seen_urls = set()

    all_matched_articles = []
    for feed_name, feed_url in RSS_FEEDS.items():
        print(f"\n--- フィードをチェック中: {feed_name} ---")
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"  -> エラー: フィードの取得に失敗しました。詳細: {e}")
            continue

        for entry in feed.entries:
            article_url = entry.link
            if article_url in seen_urls:
                continue
            
            article_title = entry.title
            article_summary = entry.get('summary', '')
            search_text = article_title + " " + article_summary
            
            for code, data in search_map.items():
                for keyword in data.get("search_keywords", []):
                    if keyword.lower() in search_text.lower():
                        print(f"  -> ヒット: [{code}] {keyword} -> {article_title}")
                        matched_article = {
                            "code": code,
                            "matched_keyword": keyword,
                            "title": article_title,
                            "url": article_url,
                            "summary": article_summary,
                            "published": entry.get('published', 'N/A'),
                            "source_feed": feed_name,
                            "collected_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        }
                        all_matched_articles.append(matched_article)
                        seen_urls.add(article_url)
                        break

    if all_matched_articles:
        new_df = pd.DataFrame(all_matched_articles)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(OUTPUT_CSV_PATH, index=False, encoding='utf-8-sig')
        print(f"\n★★★ 新しく {len(new_df)}件のヘッドラインを追加し、合計 {len(combined_df)} 件になりました ★★★")
    else:
        print("\n--- 新しくヒットするヘッドラインは見つかりませんでした ---")


if __name__ == "__main__":
    collect_headlines()