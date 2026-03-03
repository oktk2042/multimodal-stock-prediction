import unicodedata
from pathlib import Path

import pandas as pd
import torch
from dateutil import parser
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 1. パスと設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 入力: RSSで収集したヘッドラインCSV
INPUT_CSV = PROJECT_ROOT / "1_data" / "processed" / "collected_headlines.csv"

# 出力: ニュース感情スコア (日次集計済み)
OUTPUT_CSV = PROJECT_ROOT / "1_data" / "processed" / "news_sentiment_features.csv"

# 学習済みモデルのパス
MODEL_PATH = PROJECT_ROOT / "4_models" / "finbert_chabsa_trained"

# 計算設定
MAX_LEN = 512
BATCH_SIZE = 16

# ==========================================
# 2. モデルロード
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"エラー: モデルが見つかりません: {MODEL_PATH}")
        print("先に train_finbert.py を実行してください。")
        exit()

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    return tokenizer, model


# ==========================================
# 3. スコア算出ロジック
# ==========================================
def get_sentiment_score_batch(texts, tokenizer, model):
    """バッチ処理で高速にスコア算出"""
    scores = []

    # バッチごとに処理
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i : i + BATCH_SIZE]

        inputs = tokenizer(batch_texts, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding=True)

        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

            # 0=Neg, 1=Neu, 2=Pos -> Score = Pos - Neg
            # モデルのconfig.id2labelに依存するが、train_finbert.pyの仕様に準拠
            current_scores = probs[:, 2] - probs[:, 0]
            scores.extend(current_scores)

    return scores


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    return text


# ==========================================
# 4. メイン処理
# ==========================================
def main():
    if not INPUT_CSV.exists():
        print(f"エラー: RSSデータが見つかりません: {INPUT_CSV}")
        return

    print("RSSデータを読み込んでいます...")
    df = pd.read_csv(INPUT_CSV)
    print(f"データ件数: {len(df)} 件")

    # テキスト結合 (タイトル + 要約)
    df["full_text"] = df["title"].fillna("") + " " + df["summary"].fillna("")
    df["full_text"] = df["full_text"].apply(normalize_text)

    # 空データを削除
    df = df[df["full_text"].str.len() > 5].copy()

    # モデルロード
    tokenizer, model = load_model()

    print("感情スコアを算出中...")
    all_texts = df["full_text"].tolist()
    scores = get_sentiment_score_batch(all_texts, tokenizer, model)

    df["News_Sentiment"] = scores

    # --- 日付の整形 ---
    # RSSの日付形式 (例: "Sun, 11 Jan 2026...") を YYYY-MM-DD に変換
    def parse_date(date_str):
        try:
            return parser.parse(date_str).strftime("%Y-%m-%d")
        except:
            return None

    df["Date"] = df["published"].apply(parse_date)
    # 変換できなかった行（今日の日付などで代用するか、削除）
    df = df.dropna(subset=["Date"])

    # --- 日次・銘柄ごとに集計 ---
    # 同一銘柄・同一日に複数ニュースがある場合、スコアを平均する
    # ※ 最大値（インパクト重視）にする手もあるが、まずは平均で。
    daily_sentiment = df.groupby(["Date", "code"])["News_Sentiment"].mean().reset_index()
    daily_sentiment.rename(columns={"code": "Code"}, inplace=True)

    # 結果保存
    daily_sentiment.to_csv(OUTPUT_CSV, index=False)

    print(f"\n完了: {OUTPUT_CSV}")
    print("--- 生成データサンプル ---")
    print(daily_sentiment.head())
    print("-" * 30)
    print(f"スコア付きニュース数: {len(daily_sentiment)} 行")


if __name__ == "__main__":
    main()
