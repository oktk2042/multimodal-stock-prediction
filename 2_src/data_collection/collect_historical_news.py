import random
import time
import urllib.parse
from pathlib import Path

import feedparser
import pandas as pd
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_FILE = DATA_DIR / "collected_news_historical_full.csv"

# 株価データファイル (ここから銘柄と期間を取得)
STOCK_FILE = DATA_DIR / "stock_data_features_v1.csv"

# 【重要】制限なし (None) に設定
TARGET_STOCK_LIMIT = None

# マクロ経済検索ワード (市場全体の動き用)
MACRO_KEYWORDS = ["日経平均 株価", "TOPIX 市況", "円相場 ドル円", "米国株 ニューヨーク市場", "日銀 金融政策"]

# Google News RSS (期間指定検索用)
# q={keyword} after:{start_date} before:{end_date}
GOOGLE_NEWS_RSS_HISTORICAL = "https://news.google.com/rss/search?q={query}&hl=ja&gl=JP&ceid=JP:ja"


# ==========================================
# 2. 関数定義
# ==========================================
def read_csv_safe(path):
    if not path.exists():
        return pd.DataFrame()
    for enc in ["utf-8", "utf-8-sig", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except:
            continue
    return pd.DataFrame()


def fetch_rss_historical(keyword, start_date, end_date, code):
    """期間を指定してRSSを取得"""
    query_str = f"{keyword} after:{start_date} before:{end_date}"
    encoded_query = urllib.parse.quote(query_str)
    url = GOOGLE_NEWS_RSS_HISTORICAL.format(query=encoded_query)

    news_list = []
    try:
        # リトライ処理
        for _ in range(3):
            try:
                feed = feedparser.parse(url)
                if hasattr(feed, "entries"):
                    break
            except:
                time.sleep(2)

        for entry in feed.entries:
            published = "Unknown"
            if hasattr(entry, "published"):
                published = entry.published
            elif hasattr(entry, "updated"):
                published = entry.updated

            try:
                dt = pd.to_datetime(published)
                pub_date = dt.strftime("%Y-%m-%d")
            except:
                # 日付パース失敗時は期間の中間日を入れる
                pub_date = start_date

            news_list.append(
                {
                    "Date": pub_date,
                    "Code": code,
                    "Keyword": keyword,
                    "Title": entry.title,
                    "Link": entry.link,
                    "Source": "GoogleNews_Historical",
                }
            )

    except Exception as e:
        print(f"Error fetching {keyword}: {e}")

    return news_list


def generate_monthly_periods(start_date, end_date):
    """開始日から終了日まで1ヶ月ごとの期間リストを生成"""
    periods = []
    current = start_date
    while current < end_date:
        next_month = current + relativedelta(months=1)
        period_end = min(next_month, end_date)

        periods.append((current.strftime("%Y-%m-%d"), period_end.strftime("%Y-%m-%d")))
        current = next_month
    return periods


def main():
    print("--- 過去ニュース収集プロセス (完全版) 開始 ---")

    # 1. 銘柄データの読み込み
    if not STOCK_FILE.exists():
        print(f"株価データが見つかりません: {STOCK_FILE}")
        return

    df_stock = read_csv_safe(STOCK_FILE)
    if df_stock.empty:
        return

    # カラム名統一
    if "Code" in df_stock.columns:
        df_stock.rename(columns={"Code": "code"}, inplace=True)
    if "Name" in df_stock.columns:
        df_stock.rename(columns={"Name": "name"}, inplace=True)
    if "Date" in df_stock.columns:
        df_stock["Date"] = pd.to_datetime(df_stock["Date"])

    # 銘柄コードを文字列化
    df_stock["code"] = df_stock["code"].astype(str)

    # 2. 各銘柄のデータ期間を取得 (ここが重要)
    # 銘柄ごとに start_date, end_date を算出
    print("各銘柄のデータ期間を計算中...")
    stock_ranges = df_stock.groupby("code").agg(
        {
            "Date": ["min", "max"],
            "name": "first",  # 社名も保持
        }
    )
    stock_ranges.columns = ["start_date", "end_date", "name"]
    stock_ranges = stock_ranges.reset_index()

    if TARGET_STOCK_LIMIT:
        print(f"※テストモード: 上位 {TARGET_STOCK_LIMIT} 銘柄のみ実行します")
        stock_ranges = stock_ranges.head(TARGET_STOCK_LIMIT)

    total_stocks = len(stock_ranges)
    print(f"収集対象: {total_stocks} 銘柄")

    # 3. 既存データの読み込み (中断再開用)
    processed_codes = set()
    existing_data = []

    if OUTPUT_FILE.exists():
        df_exist = read_csv_safe(OUTPUT_FILE)
        if not df_exist.empty:
            # 既に処理が終わった銘柄コードを特定
            # マクロニュース(9999)は毎回更新チェックするため除外しない
            processed_codes = set(df_exist[df_exist["Code"] != "9999"]["Code"].astype(str).unique())
            existing_data = df_exist.to_dict("records")
            print(f"再開モード: 既に {len(processed_codes)} 銘柄の収集が完了しています。スキップします。")

    # リスト初期化
    new_data = []

    try:
        # --- Phase 1: マクロニュース (未実施の場合のみ、あるいは更新) ---
        # 全銘柄共通の期間 (全体のMin〜Max)
        overall_min = df_stock["Date"].min()
        overall_max = df_stock["Date"].max()
        macro_periods = generate_monthly_periods(overall_min, overall_max)

        # マクロは毎回やるか、フラグ管理するかだが、ここでは軽量なので実施
        print(f"\n[Phase 1] マクロ経済ニュース ({len(macro_periods)} 期間)")
        for keyword in MACRO_KEYWORDS:
            # 既にデータがある期間はスキップするロジックを入れてもいいが、
            # 複雑になるので上書き更新の方針で行く
            for start_d, end_d in tqdm(macro_periods, desc=f"Macro: {keyword}", leave=False):
                res = fetch_rss_historical(keyword, start_d, end_d, code="9999")
                new_data.extend(res)
                time.sleep(1.0 + random.random())  # ランダムWait

        # 中間保存
        save_intermediate(existing_data, new_data)
        existing_data.extend(new_data)
        new_data = []

        # --- Phase 2: 個別銘柄ニュース ---
        print(f"\n[Phase 2] 個別銘柄ニュース ({len(stock_ranges)} 社)")

        # イテレーション
        for idx, row in tqdm(stock_ranges.iterrows(), total=len(stock_ranges), desc="Progress"):
            code = str(row["code"])
            name = str(row["name"])
            start_d = row["start_date"]
            end_d = row["end_date"]

            # 既に完了していればスキップ
            if code in processed_codes:
                continue

            # その銘柄の期間分だけ月次ループを作る
            periods = generate_monthly_periods(start_d, end_d)

            # 検索クエリ: "トヨタ自動車 決算" のように絞るか、社名単体か
            # 社名単体だとノイズが多いが、取りこぼしは減る。
            # ここでは「社名」で広く取る
            search_query = f"{name}"

            stock_news = []
            for s_date, e_date in periods:
                res = fetch_rss_historical(search_query, s_date, e_date, code=code)
                stock_news.extend(res)
                # Googleへの負荷を考慮してWaitを入れる
                time.sleep(1.5 + random.random())

            new_data.extend(stock_news)

            # 1銘柄終わるごとに保存 (これが重要)
            save_intermediate(existing_data, new_data)
            existing_data.extend(new_data)
            new_data = []

            # 完了リストに追加
            processed_codes.add(code)

    except KeyboardInterrupt:
        print("\n中断されました。直前のデータを保存します...")
        save_intermediate(existing_data, new_data)
        print("保存完了。再実行すると続きから始まります。")
        return

    print("\n全収集完了しました。")


def save_intermediate(existing, new_items):
    """データをCSVに上書き保存"""
    if not new_items and not existing:
        return

    # メモリ節約のため、Dataframe操作は保存時のみ
    df_new = pd.DataFrame(new_items)

    if existing:
        df_exist = pd.DataFrame(existing)
        df_final = pd.concat([df_exist, df_new], ignore_index=True)
    else:
        df_final = df_new

    # 重複排除
    if not df_final.empty:
        df_final = df_final.drop_duplicates(subset=["Date", "Code", "Title"])
        df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
