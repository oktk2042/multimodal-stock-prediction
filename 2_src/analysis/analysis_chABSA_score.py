import json
from pathlib import Path

# ==========================================
# 設定
# ==========================================
# JSONファイルが格納されているディレクトリ
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "chABSA" / "data" / "annotated"


def extract_chabsa_examples():
    json_files = list(DATA_DIR.glob("e*_ann.json"))
    print(f"検索対象ファイル数: {len(json_files)}")
    print("-" * 50)

    found_count = 0

    for filepath in json_files:
        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        for sent in data["sentences"]:
            ops = sent["opinions"]
            if not ops:
                continue

            # 極性のセットを取得
            polarities = {op["polarity"] for op in ops}

            # 【条件】PositiveとNegativeが両方含まれる「難しい文」を探す
            if "positive" in polarities and "negative" in polarities:
                print(f"File: {filepath.name}")
                print(f"文: {sent['sentence']}")
                print("アノテーション:")
                for op in ops:
                    print(f"  - Target: {op['target']:<15} | Polarity: {op['polarity']}")
                print("-" * 50)

                found_count += 1
                if found_count >= 5:  # 5件見つかったら終了
                    return


if __name__ == "__main__":
    extract_chabsa_examples()
