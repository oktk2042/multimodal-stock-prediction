import os

# 除外したいディレクトリ名（完全一致）
# ここに含まれるフォルダは、フォルダ名のみ表示し、中のファイルは表示しません
IGNORE_DIRS = {".git", ".idea", "__pycache__", "venv", ".venv", "node_modules", ".mypy_cache", ".ruff_cache"}

# 出力ファイル名
OUTPUT_FILE = "project_structure.txt"


def generate_tree(startpath):
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for root, dirs, files in os.walk(startpath):
            # ルートからの深さを計算
            level = root.replace(startpath, "").count(os.sep)
            indent = "    " * level
            dirname = os.path.basename(root)

            # 除外リストに含まれるディレクトリの場合、その配下の探索を止める処理
            if dirname in IGNORE_DIRS:
                f.write(f"{indent}{dirname}/ (*** 省略 ***)\n")
                dirs[:] = []  # これより下層は探索しない
                continue

            # ディレクトリ名の書き込み（ルート以外）
            if root != startpath:
                f.write(f"{indent}{dirname}/\n")
            else:
                f.write(f"{dirname}/\n")

            # ファイル名の書き込み
            subindent = "    " * (level + 1)
            for file in files:
                # .DS_Storeなどのシステムファイルやpycファイルは除外して見やすくする
                if file == ".DS_Store" or file.endswith(".pyc"):
                    continue
                f.write(f"{subindent}{file}\n")


if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"ディレクトリ構造を取得中...: {current_dir}")
    generate_tree(current_dir)
    print(f"完了しました。 '{OUTPUT_FILE}' を確認してください。")
