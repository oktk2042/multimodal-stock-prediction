import os
import glob
import pandas as pd
import torch
import numpy as np
import unicodedata
from tqdm import tqdm
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. パスと設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 入力ディレクトリ
INPUT_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

# 出力ファイル
OUTPUT_FILE = PROJECT_ROOT / "1_data" / "processed" / "edinet_features_finbert.csv"

# 学習済みモデルのパス
MODEL_PATH = PROJECT_ROOT / "4_models" / "finbert_chabsa_trained"

# ---【重要修正】計算設定---
# 学習時のモデル設定(max_position_embeddings)に合わせる
CHUNK_SIZE = 128  
STRIDE = 64       
BATCH_SIZE = 64   

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
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH), use_safetensors=True)
        model = AutoModelForSequenceClassification.from_pretrained(str(MODEL_PATH), use_safetensors=True)

    model.to(device)
    model.eval()
    return tokenizer, model

# ==========================================
# 3. テキスト処理ロジック
# ==========================================
def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\n", "").replace("\r", "").replace("\t", " ")
    return text

def read_csv_robust(file_path):
    """
    共有いただいたデータ形式(UTF-8, Tab区切り)に対応するための堅牢な読み込み
    """
    # 優先順位: 共有データはUTF-8だったので先頭に
    encodings = ['utf-8', 'utf-16', 'utf-16-le', 'cp932', 'shift_jis']
    
    # 優先順位: 共有データはTab区切りだったので先頭に
    separators = ['\t', ',']
    
    for enc in encodings:
        for sep in separators:
            try:
                # engine='python'で柔軟に読み込む
                df = pd.read_csv(file_path, encoding=enc, sep=sep, on_bad_lines='skip')
                
                # 正常に読めたかのチェック (カラムが1つ以上あるか)
                if not df.empty and len(df.columns) > 1:
                    return df
            except:
                continue
                
    # どうしても読めない場合はNone
    return None

def load_text_from_xbrl_csv(doc_folder_path):
    """
    DocIDフォルダ内のCSVからテキストを抽出・結合する。
    「値」カラムを優先的に探す。
    """
    text_content = ""
    
    # ターゲットフォルダ: DocID/XBRL_TO_CSV (なければ再帰探索)
    target_dir = doc_folder_path / "XBRL_TO_CSV"
    if not target_dir.exists():
        csv_files = list(doc_folder_path.rglob("*.csv"))
    else:
        csv_files = list(target_dir.glob("*.csv"))
    
    if not csv_files:
        return ""

    for csv_file in csv_files:
        df = read_csv_robust(csv_file)
        if df is None:
            continue
            
        try:
            # 共有データのカラム名「値」を優先して取得
            target_col = None
            if '値' in df.columns:
                target_col = '値'
            elif 'Value' in df.columns:
                target_col = 'Value'
            
            if target_col:
                # 特定カラムがある場合、その中のテキストだけを取得 (ノイズが減る)
                values = df[target_col].astype(str).values
            else:
                # 見つからない場合は全カラムをフラットに (フォールバック)
                values = df.astype(str).values.flatten()
            
            for val in values:
                # フィルタリング:
                # 1. ある程度の長さがある (30文字以上) -> 数値データや短いIDを除外
                # 2. 'nan' ではない
                if len(val) > 30 and val.lower() != 'nan':
                    cleaned = normalize_text(val)
                    text_content += cleaned + " "
                        
        except Exception:
            continue
            
    return text_content.strip()

def get_sentiment_score(text, tokenizer, model):
    """
    長文対応: 128トークンごとに分割し、スコアを算出・平均化する
    """
    if len(text) < 30: 
        return None 

    # トークナイズ
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=CHUNK_SIZE, # 128
        truncation=True,
        padding="max_length",
        stride=STRIDE,         # 64
        return_overflowing_tokens=True
    )

    input_ids_all = inputs['input_ids']
    attention_mask_all = inputs['attention_mask']
    
    chunk_scores = []
    
    with torch.no_grad():
        # バッチ処理
        for i in range(0, len(input_ids_all), BATCH_SIZE):
            input_ids = input_ids_all[i : i + BATCH_SIZE].to(device)
            attention_mask = attention_mask_all[i : i + BATCH_SIZE].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()
            
            # 0=Neg, 1=Neu, 2=Pos -> Score = Pos - Neg
            current_scores = probs[:, 2] - probs[:, 0]
            chunk_scores.extend(current_scores)

    if not chunk_scores:
        return None

    return float(np.round(np.mean(chunk_scores), 4))

# ==========================================
# 4. メイン実行ループ
# ==========================================
def main():
    if not INPUT_DIR.exists():
        print(f"エラー: 入力ディレクトリが見つかりません: {INPUT_DIR}")
        return

    tokenizer, model = load_model()

    # S100... で始まるフォルダを取得
    doc_folders = [p for p in INPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
    print(f"探索ディレクトリ: {INPUT_DIR}")
    print(f"処理対象レポート数 (S100*): {len(doc_folders)}")
    
    results = []
    
    # tqdmで進捗表示
    for folder in tqdm(doc_folders, desc="FinBERT Scoring"):
        doc_id = folder.name
        
        full_text = load_text_from_xbrl_csv(folder)
        
        # テキストが取れなかった場合はスキップ
        if not full_text:
            continue
            
        score = get_sentiment_score(full_text, tokenizer, model)
        
        if score is not None:
            results.append({
                "DocID": doc_id,
                "FinBERT_Score": score,
                "TextLength": len(full_text)
            })
            
    if results:
        df_result = pd.DataFrame(results)
        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_result.to_csv(OUTPUT_FILE, index=False)
        print(f"\n完了: {OUTPUT_FILE}")
        print(f"抽出件数: {len(df_result)}")
        print(df_result.head())
    else:
        print("\n有効なテキストデータが見つかりませんでした。")
        print("csvファイルの中身を確認してください。")

if __name__ == "__main__":
    main()