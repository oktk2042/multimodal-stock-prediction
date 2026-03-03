import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ==========================================
# 1. パスと設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 入力ディレクトリ (ZIPファイルがある場所)
ZIP_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "01_zip_files_indices"

# 出力ファイル (新規作成)
OUTPUT_FILE = PROJECT_ROOT / "1_data" / "processed" / "edinet_features_finbert_indices_strict.csv"

# 学習済みモデルのパス
MODEL_PATH = PROJECT_ROOT / "4_models" / "finbert_chabsa_trained"

# 計算設定
CHUNK_SIZE = 126
STRIDE = 64
BATCH_SIZE = 32

# ==========================================
# 2. モデルロード
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        print(f"エラー: モデルが見つかりません: {MODEL_PATH}")
        return None, None

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    return tokenizer, model


# ==========================================
# 3. テキスト抽出 (ZIP内のHTMLから)
# ==========================================
def extract_text_from_zip(zip_path):
    """ZIP内の主要なHTMLファイルからテキストを抽出・結合する"""
    text_content = ""
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            files = [f for f in zf.namelist() if f.endswith(".htm") and "PublicDoc" in f]
            files.sort()

            full_text = []
            for filename in files:
                with zf.open(filename) as f:
                    soup = BeautifulSoup(f.read(), "lxml")
                    for script in soup(["script", "style"]):
                        script.decompose()
                    text = soup.get_text(separator=" ", strip=True)
                    if len(text) > 100:
                        full_text.append(text)

            text_content = " ".join(full_text)

    except Exception:
        pass

    return text_content


# ==========================================
# 4. 厳密なスコアリング関数 (-1 ~ 1)
# ==========================================
def get_sentiment_score_strict(text, tokenizer, model):
    if not text:
        return None

    tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    total_len = len(tokens)

    if total_len == 0:
        return None

    chunk_scores = []

    # チャンクごとに推論
    for i in range(0, total_len, STRIDE):
        chunk = tokens[i : i + CHUNK_SIZE]
        if len(chunk) < 10:
            continue

        input_ids = tokenizer.build_inputs_with_special_tokens(chunk.tolist())
        input_tensor = torch.tensor([input_ids]).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            # probs shape: [1, 3] -> [Neg, Neu, Pos] (chabsaの場合)

            neg_prob = probs[0][0].item()
            neu_prob = probs[0][1].item()
            pos_prob = probs[0][2].item()

            # 厳密なスコア計算 (-1 ~ 1)
            # Positive - Negative で算出 (Neutralは無視、または0として扱う)
            # 例: Pos=0.8, Neg=0.1 -> Score = 0.7
            # 例: Pos=0.1, Neg=0.8 -> Score = -0.7
            score = pos_prob - neg_prob

            chunk_scores.append(score)

    if not chunk_scores:
        return None

    # ドキュメント全体の平均スコア
    return np.mean(chunk_scores)


# ==========================================
# 5. メイン処理
# ==========================================
def main():
    if not ZIP_DIR.exists():
        print(f"エラー: ディレクトリなし {ZIP_DIR}")
        return

    tokenizer, model = load_model()
    if model is None:
        return

    zip_files = list(ZIP_DIR.glob("*.zip"))
    print(f"処理対象: {len(zip_files)} 件")

    # レジューム機能
    processed_ids = set()
    if OUTPUT_FILE.exists():
        try:
            df_exist = pd.read_csv(OUTPUT_FILE)
            processed_ids = set(df_exist["DocID"].unique())
            print(f"処理済み: {len(processed_ids)} 件 (スキップ)")
        except Exception:
            pass

    results = []

    for i, zip_file in enumerate(tqdm(zip_files, desc="Strict FinBERT Scoring")):
        doc_id = zip_file.stem
        if doc_id in processed_ids:
            continue

        full_text = extract_text_from_zip(zip_file)

        if len(full_text) > 0:
            # ここで厳密版を呼ぶ
            score = get_sentiment_score_strict(full_text, tokenizer, model)

            if score is not None:
                results.append({"DocID": doc_id, "FinBERT_Score": score, "TextLength": len(full_text)})

        # 50件ごとに保存
        if len(results) >= 50:
            df_chunk = pd.DataFrame(results)
            if not OUTPUT_FILE.exists():
                df_chunk.to_csv(OUTPUT_FILE, index=False)
            else:
                df_chunk.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
            processed_ids.update(df_chunk["DocID"].tolist())
            results = []

    # 残り保存
    if results:
        df_chunk = pd.DataFrame(results)
        if not OUTPUT_FILE.exists():
            df_chunk.to_csv(OUTPUT_FILE, index=False)
        else:
            df_chunk.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)

    print(f"完了: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
