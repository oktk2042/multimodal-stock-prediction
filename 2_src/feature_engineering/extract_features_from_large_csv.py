from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル (今収集しているファイル)
INPUT_FILE = DATA_DIR / "collected_news_historical_full.csv"

# 出力ファイル
OUTPUT_FILE = DATA_DIR / "news_sentiment_historical.csv"

# モデルパス
MODEL_PATH = PROJECT_ROOT / "4_models" / "finbert_chabsa_trained"

# 計算設定
BATCH_SIZE = 64  # GPUメモリに合わせて調整 (RTX 4070なら64-128いけるかも)
MAX_LEN = 128  # 学習時に合わせた長さ

# ==========================================
# 2. モデルロード
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    model.to(device)
    model.eval()
    return tokenizer, model


def get_sentiment_batch(texts, tokenizer, model):
    """バッチ処理で高速にスコア算出"""
    # 前処理
    clean_texts = [str(t).replace("\n", "")[:200] for t in texts]  # 長すぎるのはカット

    inputs = tokenizer(clean_texts, return_tensors="pt", max_length=MAX_LEN, truncation=True, padding="max_length")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

    # Score = Positive - Negative
    scores = probs[:, 2] - probs[:, 0]
    return scores


# ==========================================
# 3. メイン処理 (チャンク読み込み)
# ==========================================
def main():
    if not INPUT_FILE.exists():
        print(f"まだファイルがありません: {INPUT_FILE}")
        print("ニュース収集が終わってから実行してください。")
        return

    print("--- 巨大ニュースデータのスコア化開始 ---")

    # 既に処理済みの行数を確認 (中断再開用)
    processed_count = 0
    if OUTPUT_FILE.exists():
        # 行数をカウント (ヘッダー除く)
        with open(OUTPUT_FILE, encoding="utf-8-sig") as f:
            processed_count = sum(1 for _ in f) - 1
        print(f"再開モード: 既に {processed_count} 行処理済みです。")

    tokenizer, model = load_model()

    # Pandasで巨大ファイルを「チャンク(塊)」ごとに読み込む
    # 1回に 10,000行 ずつ処理する
    chunksize = 10000

    # 全体の行数を取得（プログレスバー用）
    # ※時間がかかるので、概算やスキップでも良いが、初回はカウント推奨
    print("全データ行数をカウント中...")
    total_rows = sum(1 for _ in open(INPUT_FILE, encoding="utf-8-sig")) - 1
    print(f"全データ: {total_rows} 行")

    # 読み込み開始
    # skiprowsを使って、既に処理した分を飛ばす
    reader = pd.read_csv(INPUT_FILE, chunksize=chunksize, encoding="utf-8-sig", skiprows=range(1, processed_count + 1))

    # 初回書き込みかどうか（ヘッダー用）
    is_first_write = processed_count == 0

    for chunk in tqdm(reader, total=(total_rows - processed_count) // chunksize, desc="Processing Chunks"):
        if chunk.empty:
            break

        # テキストがない行は除外
        chunk["Title"] = chunk["Title"].fillna("")
        valid_indices = chunk[chunk["Title"].str.len() > 1].index

        if len(valid_indices) == 0:
            continue

        texts = chunk.loc[valid_indices, "Title"].tolist()

        # バッチ処理で推論 (さらに細かく分割してGPUに投げる)
        scores = []
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_scores = get_sentiment_batch(batch_texts, tokenizer, model)
            scores.extend(batch_scores)

        # 結果を格納
        chunk.loc[valid_indices, "News_Sentiment"] = scores

        # 必要なカラムだけ保存
        output_df = chunk[["Date", "Code", "Title", "News_Sentiment", "Keyword", "Source"]].copy()

        # 追記モード('a')で保存
        output_df.to_csv(OUTPUT_FILE, mode="a", header=is_first_write, index=False, encoding="utf-8-sig")
        is_first_write = False

    print("\n完了しました！")
    print(f"保存先: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
