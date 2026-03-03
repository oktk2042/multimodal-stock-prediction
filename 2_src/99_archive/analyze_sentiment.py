import pandas as pd
import os
import google.generativeai as genai
import json
import time
from tqdm import tqdm # 進捗バー表示用

# --- 設定 ---
# Google AI StudioのAPIキー
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
genai.configure(api_key=GOOGLE_API_KEY)

# 入力・出力ディレクトリ
RAW_NEWS_DIR = "1_data/raw/news/"
PROCESSED_NEWS_DIR = "1_data/processed/news_with_sentiment/"
os.makedirs(PROCESSED_NEWS_DIR, exist_ok=True)

def analyze_sentiment_with_llm():
    """
    収集したニュース記事を読み込み、LLMで感情スコアを付与する
    """
    model = genai.GenerativeModel('gemini-pro')
    
    news_files = [f for f in os.listdir(RAW_NEWS_DIR) if f.endswith('_news.csv')]
    
    for filename in news_files:
        print(f"--- {filename} の感情分析を開始 ---")
        input_path = os.path.join(RAW_NEWS_DIR, filename)
        output_path = os.path.join(PROCESSED_NEWS_DIR, filename.replace('_news.csv', '_with_sentiment.csv'))

        if os.path.exists(output_path):
            print(" -> 既に分析済みファイルが存在するため、スキップします。")
            continue

        df = pd.read_csv(input_path)
        df = df.dropna(subset=['title', 'description']) # タイトルと概要がない記事は除外

        results = []
        # tqdmを使って進捗バーを表示
        for index, row in tqdm(df.iterrows(), total=df.shape[0]):
            try:
                prompt = f"""
                あなたは金融ニュースを分析する専門家です。
                以下のニュース記事のタイトルと概要を読んで、この記事が企業の株価に対して「ポジティブ」「ニュートラル」「ネガティブ」のどれに該当するかを判断してください。
                結果は必ず以下のJSON形式で、sentimentとscoreの2つのキーで出力してください。
                scoreはポジティブを1、ニュートラルを0、ネガティブを-1とします。

                タイトル: {row['title']}
                概要: {row['description']}

                {{
                  "sentiment": "（ここにポジティブ/ニュートラル/ネガティブのいずれか）",
                  "score": （ここに1/0/-1のいずれか）
                }}
                """
                
                response = model.generate_content(prompt)
                json_response = json.loads(response.text.strip().replace("```json", "").replace("```", ""))
                
                results.append({
                    'sentiment_label': json_response['sentiment'],
                    'sentiment_score': json_response['score']
                })

            except Exception as e:
                print(f"エラー: 記事No.{index}の分析中に問題 - {e}")
                results.append({'sentiment_label': 'error', 'sentiment_score': 0}) # エラー時は0
            
            time.sleep(1) # APIのレート制限を考慮

        # 元のデータに分析結果を結合
        result_df = pd.DataFrame(results)
        df_with_sentiment = pd.concat([df, result_df], axis=1)
        
        df_with_sentiment.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f" -> 分析結果を {output_path} に保存しました。")


if __name__ == "__main__":
    analyze_sentiment_with_llm()
    print("\n★★★ 全てのニュースの感情分析が完了しました ★★★")