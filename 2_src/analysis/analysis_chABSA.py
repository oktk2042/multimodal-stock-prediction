import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ==========================================
# 設定
# ==========================================
# JSONファイルが格納されているディレクトリ
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "chABSA" / "data" / "annotated"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# グラフのスタイル設定
plt.rcParams["font.family"] = "MS Gothic"  # 日本語フォントがあれば指定推奨 (例: MS Gothic)
sns.set(style="whitegrid")


def analyze_chabsa():
    json_files = list(DATA_DIR.glob("e*_ann.json"))
    print(f"Found {len(json_files)} files.")

    stats = []
    opinions_list = []
    mixed_sentiment_sentences = []

    for filepath in json_files:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)

        doc_sentences = 0
        doc_opinions = 0

        for sent in data["sentences"]:
            doc_sentences += 1
            text = sent["sentence"]
            ops = sent["opinions"]

            # 意見がない文はスキップ（学習に使わない場合）
            if not ops:
                continue

            doc_opinions += len(ops)

            # 1文内の極性セットを作成
            polarities = set()

            for op in ops:
                opinions_list.append(
                    {
                        "doc_id": data["header"]["document_id"],
                        "sentence_id": sent["sentence_id"],
                        "target": op["target"],
                        "category": op["category"],
                        "polarity": op["polarity"],
                    }
                )
                polarities.add(op["polarity"])

            # ポジティブとネガティブが混在する「難しい文」を抽出
            if "positive" in polarities and "negative" in polarities:
                mixed_sentiment_sentences.append(
                    {"text": text, "opinions": ops, "doc_id": data["header"]["document_id"]}
                )

        stats.append({"doc_id": data["header"]["document_id"], "sentences": doc_sentences, "opinions": doc_opinions})

    # --- 1. 基本統計量の集計 ---
    df_ops = pd.DataFrame(opinions_list)
    total_docs = len(json_files)
    total_sentences = sum(s["sentences"] for s in stats)
    total_opinions = len(df_ops)

    print("=" * 30)
    print(" chABSA Dataset Statistics ")
    print("=" * 30)
    print(f"Total Documents: {total_docs}")
    print(f"Total Sentences: {total_sentences}")
    print(f"Total Opinions : {total_opinions}")
    print(f"Mixed Sentiment Sentences: {len(mixed_sentiment_sentences)}")
    print("-" * 20)
    print(df_ops["polarity"].value_counts())

    # --- 2. グラフ作成 (Polarity Distribution) ---
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x="polarity", data=df_ops, order=["positive", "negative", "neutral"], palette="viridis")
    plt.title("Distribution of Sentiment Polarities in chABSA", fontsize=14)
    plt.xlabel("Polarity", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # 数値をグラフ上に表示
    for p in ax.patches:
        ax.annotate(
            f"{int(p.get_height())}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="baseline",
            fontsize=11,
            color="black",
            xytext=(0, 5),
            textcoords="offset points",
        )

    save_path = OUTPUT_DIR / "chabsa_polarity_distribution.png"
    plt.savefig(save_path, dpi=300)
    print(f"\nGraph saved to: {save_path}")

    # --- 3. 複雑な文の例を出力（論文引用などのため）---
    print("\n[Complex Sentence Example (Pos & Neg in one sentence)]")
    if mixed_sentiment_sentences:
        example = mixed_sentiment_sentences[0]  # 最初の1件を表示
        print(f"DocID: {example['doc_id']}")
        print(f"Text : {example['text']}")
        print(f"Ops  : {json.dumps(example['opinions'], ensure_ascii=False, indent=2)}")

    return df_ops, mixed_sentiment_sentences


if __name__ == "__main__":
    analyze_chabsa()
