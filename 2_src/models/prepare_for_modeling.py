from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# --- 0. 設定 ---
class Config:
    BASE_PATH = Path("C:/M2_Research_Project/1_data")
    PROCESSED_PATH = BASE_PATH / "processed"
    MODELING_PATH = BASE_PATH / "modeling_data"

    # [変更] 入力ファイル名を明示的に指定
    INPUT_CSV_PATH = PROCESSED_PATH / "final_model_input_dataset.csv"

    # v4のデータファイルを出力
    OUTPUT_TRAIN_PATH = MODELING_PATH / "train_data_v4.npz"
    OUTPUT_VALID_PATH = MODELING_PATH / "validation_data_v4.npz"
    OUTPUT_TEST_PATH = MODELING_PATH / "test_data_v4.npz"
    SCALER_PATH = MODELING_PATH / "feature_scaler_v4.gz"

    TARGET_COLUMN = "Close"

    TRAIN_END_DATE = "2023-12-31"
    VALID_END_DATE = "2024-06-30"


# --- 1. メイン実行関数 ---
def main():
    Config.MODELING_PATH.mkdir(exist_ok=True)
    print(f"--- データ読み込み開始: {Config.INPUT_CSV_PATH} ---")
    df = pd.read_csv(Config.INPUT_CSV_PATH, parse_dates=["Date"])
    df.sort_values(by=["Code", "Date"], inplace=True)

    # --- [バグ修正の核心] ---
    # 1. スケーリング対象となる数値型の列を自動的に選択
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

    # 2. その中から、モデルの入力に直接使わない'Code'などを除外
    #    'Close'はスケーラーに学習させるため、この段階では残す
    cols_to_scale = [col for col in numeric_cols if col not in ["Code"]]

    # 3. モデルの入力特徴量(X)と正解データ(y)を定義
    #    入力特徴量は、スケーリング対象から'Close'を除いたもの
    feature_cols = [col for col in cols_to_scale if col != Config.TARGET_COLUMN]

    print(f"検出された入力特徴量数: {len(feature_cols)}")

    # --- 2. スケーリング ---
    scaler = MinMaxScaler()

    # 訓練データ期間のみでスケーラーを学習させる
    train_df = df[df["Date"] <= Config.TRAIN_END_DATE]
    # [重要] 'Close'を含む全ての数値列でスケーラーを学習
    scaler.fit(train_df[cols_to_scale])

    # データフレーム全体をスケーリング
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    print("--- データのスケーリング完了 ---")

    # --- 3. データ分割 ---
    train_data = df[df["Date"] <= Config.TRAIN_END_DATE]
    valid_data = df[(df["Date"] > Config.TRAIN_END_DATE) & (df["Date"] <= Config.VALID_END_DATE)]
    test_data = df[df["Date"] > Config.VALID_END_DATE]

    # --- 4. NPZ形式で保存 ---
    def save_as_npz(data, path):
        # [修正] ここでは、事前に定義したfeature_colsを使用
        features = data[feature_cols].values.astype(np.float32)
        target = data[Config.TARGET_COLUMN].values.astype(np.float32)
        codes = data["Code"].values

        np.savez_compressed(path, features=features, target=target, codes=codes)
        print(f"{path} を保存しました。特徴量数: {features.shape[1]}")

    save_as_npz(train_data, Config.OUTPUT_TRAIN_PATH)
    save_as_npz(valid_data, Config.OUTPUT_VALID_PATH)
    save_as_npz(test_data, Config.OUTPUT_TEST_PATH)

    # --- 5. スケーラーの保存 ---
    joblib.dump(scaler, Config.SCALER_PATH)
    print(f"--- スケーラーを保存しました: {Config.SCALER_PATH} ---")
    print(f"スケーラーが学習した特徴量: {scaler.feature_names_in_}")


if __name__ == "__main__":
    main()
