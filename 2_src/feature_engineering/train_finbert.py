import glob
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# ==========================================
# 1. パスと設定（ディレクトリ変更に対応）
# ==========================================
# このファイルの2つ上の階層をプロジェクトルートとする
# 2_src/feature_engineering/train_finbert.py -> parents[2] = M2_Research_Project
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# chABSAデータのパス
DATA_DIR = PROJECT_ROOT / "1_data" / "chABSA" / "data" / "annotated"

# モデル保存先
OUTPUT_DIR = PROJECT_ROOT / "4_models" / "finbert_chabsa_trained"

# ベースモデル (東大・和泉研の金融特化モデル)
BASE_MODEL = "izumi-lab/bert-small-japanese-fin"

# ハイパーパラメータ
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5
SEED = 42

set_seed(SEED)


# ==========================================
# 2. データ読み込み関数
# ==========================================
def load_chabsa_data(data_dir):
    sentences = []
    labels = []

    # Pathオブジェクトを文字列に変換してglobに使用
    files = glob.glob(str(data_dir / "**/*.json"), recursive=True)
    print(f"探索ディレクトリ: {data_dir}")
    print(f"発見ファイル数: {len(files)}")

    if len(files) == 0:
        raise FileNotFoundError(f"データが見つかりません。パスを確認してください: {data_dir}")

    for file_path in files:
        try:
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            for sent_obj in data["sentences"]:
                text = sent_obj["sentence"]
                opinions = sent_obj["opinions"]

                if not opinions:
                    continue

                score = 0
                for op in opinions:
                    pol = op["polarity"]
                    if pol == "positive":
                        score += 1
                    elif pol == "negative":
                        score -= 1

                # 0=Negative, 1=Neutral, 2=Positive
                if score > 0:
                    label = 2
                elif score < 0:
                    label = 0
                else:
                    label = 1

                sentences.append(text)
                labels.append(label)

        except Exception as e:
            print(f"Skipping {os.path.basename(file_path)}: {e}")
            continue

    return sentences, labels


# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print(f"Project Root: {PROJECT_ROOT}")

    # 1. データ読み込み
    texts, labels = load_chabsa_data(DATA_DIR)
    print(f"Loaded {len(texts)} sentences.")

    if len(texts) == 0:
        return

    # 2. データ分割 (Train/Val)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=SEED, stratify=labels
    )

    # 3. Dataset作成
    train_ds = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_ds = Dataset.from_dict({"text": val_texts, "label": val_labels})

    # 4. トークナイザ
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=MAX_LEN)

    tokenized_train = train_ds.map(preprocess_function, batched=True)
    tokenized_val = val_ds.map(preprocess_function, batched=True)

    # 5. モデル準備
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2},
        use_safetensors=True,
    )

    # 6. 学習設定
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        logging_dir=str(OUTPUT_DIR / "logs"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    print("--- 学習開始 ---")
    trainer.train()

    print(f"--- モデル保存: {OUTPUT_DIR} ---")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))
    print("完了しました。")


if __name__ == "__main__":
    main()
