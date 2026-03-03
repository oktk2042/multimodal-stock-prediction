import os

import pandas as pd

# ==========================================
# 1. 設定
# ==========================================

# 対象のディレクトリパス
TARGET_DIR = r"C:\M2_Research_Project\3_reports\phase3_production_deep_strict"

# 【CSVの中身】用の置換マップ (正式名称・表示用)
CONTENT_MAP = {
    "Ridge": "Ridge Regression",
    "LSTM": "Attention-LSTM",
    "Transformer": "Vanilla Transformer",
    "FusionTransformer": "Multi-Modal Gated Transformer (Ours)",
    "LightGBM": "LightGBM",  # 変更なしだが念のため
    "DLinear": "DLinear",
    "PatchTST": "PatchTST",
    "iTransformer": "iTransformer",
}

# 【ファイル名】用の置換マップ (システム的に安全な名称)
# ※ Transformerが他の名前に含まれるため、置換順序が重要です
FILENAME_MAP = {
    # 優先度の高いもの（長い文字列）から定義
    "FusionTransformer": "MultiModalGatedTransformer",
    "iTransformer": "iTransformer",  # 変更しないが、Vanillaの誤爆を防ぐため定義
    "PatchTST": "PatchTST",
    "Transformer": "VanillaTransformer",  # "Fusion"や"i"が含まれていない純粋なTransformerのみ対象
    "LSTM": "Attention-LSTM",
    "Ridge": "RidgeRegression",
    "DLinear": "DLinear",
}

# ==========================================
# 2. 処理ロジック
# ==========================================


def update_csv_content(filepath):
    """CSVファイル内のModel列を正式名称に書き換える"""
    try:
        df = pd.read_csv(filepath)
        changed = False

        # 'Model' または 'Model_Display' カラムがある場合、値を置換
        target_cols = ["Model", "Model_Display"]
        for col in target_cols:
            if col in df.columns:
                # マップに基づいて置換
                for old_name, new_name in CONTENT_MAP.items():
                    # 完全一致で置換 (部分一致だと危険なため)
                    if old_name in df[col].values:
                        mask = df[col] == old_name
                        df.loc[mask, col] = new_name
                        changed = True

        if changed:
            df.to_csv(filepath, index=False)
            print(f"[Content Updated] {os.path.basename(filepath)}")

    except Exception as e:
        print(f"[Error Reading CSV] {filepath}: {e}")


def rename_file(filepath, filename):
    """ファイル名を正式名称ルールに基づいて変更する"""
    new_filename = filename

    # --- ルールベースの置換 ---
    # 1. FusionTransformer -> MultiModalGatedTransformer
    if "FusionTransformer" in new_filename:
        new_filename = new_filename.replace("FusionTransformer", FILENAME_MAP["FusionTransformer"])

    # 2. iTransformer -> (変更なし、Vanilla Transformerへの誤変換防止)
    elif "iTransformer" in new_filename:
        pass

    # 3. PatchTST -> (変更なし)
    elif "PatchTST" in new_filename:
        pass

    # 4. Vanilla Transformer (純粋なTransformer)
    elif "Transformer" in new_filename:
        new_filename = new_filename.replace("Transformer", FILENAME_MAP["Transformer"])

    # 5. LSTM -> Attention-LSTM
    if "LSTM" in new_filename:
        new_filename = new_filename.replace("LSTM", FILENAME_MAP["LSTM"])

    # 6. Ridge -> RidgeRegression
    if "Ridge" in new_filename and "RidgeRegression" not in new_filename:
        new_filename = new_filename.replace("Ridge", FILENAME_MAP["Ridge"])

    # 変更があった場合のみリネーム実行
    if new_filename != filename:
        new_path = os.path.join(os.path.dirname(filepath), new_filename)
        try:
            os.rename(filepath, new_path)
            print(f"[Renamed] {filename} -> {new_filename}")
        except OSError as e:
            print(f"[Error Renaming] {filename}: {e}")


def main():
    if not os.path.exists(TARGET_DIR):
        print(f"Directory not found: {TARGET_DIR}")
        return

    print("--- Starting Processing ---")

    # ファイルリスト取得
    files = os.listdir(TARGET_DIR)

    # 1. まずCSVの中身を更新 (リネーム前に行う)
    for filename in files:
        if filename.endswith(".csv"):
            update_csv_content(os.path.join(TARGET_DIR, filename))

    # 2. ファイル名を変更
    # リストを再取得する必要はないが、リネーム順序の影響を受けないよう注意
    for filename in files:
        rename_file(os.path.join(TARGET_DIR, filename), filename)

    print("--- Completed ---")
    print("※注意: PNG画像内のグラフタイトル文字はこのスクリプトでは変更されません。")
    print("      グラフを更新するには、描画コードを再実行してください。")


if __name__ == "__main__":
    main()
