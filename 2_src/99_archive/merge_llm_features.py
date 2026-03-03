import pandas as pd
import os

# --- 設定 ---
PROCESSED_DATA_DIR = "1_data/processed/"
SENTIMENT_NEWS_DIR = os.path.join(PROCESSED_DATA_DIR, "news_with_sentiment/")

# 入力ファイル
BASE_DATA_FILE = os.path.join(PROCESSED_DATA_DIR, "stock_data_features_v2.csv")
# 出力ファイル
FINAL_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "stock_data_final_features.csv")

def merge_daily_sentiment():
    """
    感情分析済みのニュースを日次で集計し、メインのデータセットにマージする
    """
    print("--- 日次感情スコアの集計と最終マージを開始 ---")
    
    sentiment_files = [f for f in os.listdir(SENTIMENT_NEWS_DIR) if f.endswith('_with_sentiment.csv')]
    if not sentiment_files:
        print("感情分析済みのファイルが見つかりません。")
        return
        
    # 全ての感情分析済みファイルを1つに結合
    all_sentiment_df = pd.concat([pd.read_csv(os.path.join(SENTIMENT_NEWS_DIR, f)) for f in sentiment_files])
    
    # 'publishedAt'を日付型に変換し、時間以下の情報を削除
    all_sentiment_df['Date'] = pd.to_datetime(all_sentiment_df['publishedAt']).dt.date
    all_sentiment_df['Date'] = pd.to_datetime(all_sentiment_df['Date']) # マージ用に再度変換
    
    # 銘柄コードと日付でグループ化し、感情スコアの平均値を計算
    daily_sentiment = all_sentiment_df.groupby(['Date', 'Code'])['sentiment_score'].mean().reset_index()
    daily_sentiment.rename(columns={'sentiment_score': 'Sentiment_Score_Mean'}, inplace=True)
    
    # メインのデータファイルを読み込む
    main_df = pd.read_csv(BASE_DATA_FILE, parse_dates=['Date'], dtype={'Code': str})
    
    # 日付と銘柄コードをキーにして、日次感情スコアをマージ
    final_df = pd.merge(main_df, daily_sentiment, on=['Date', 'Code'], how='left')
    
    # ニュースがなかった日の感情スコアは0（ニュートラル）で埋める
    final_df['Sentiment_Score_Mean'].fillna(0, inplace=True)
    
    # 最終的な特徴量データセットを保存
    final_df.to_csv(FINAL_OUTPUT_FILE, index=False, encoding='utf-8-sig')
    print(f" -> 完了！LLM特徴量を含む最終データセットを {FINAL_OUTPUT_FILE} に保存しました。")


if __name__ == "__main__":
    merge_daily_sentiment()