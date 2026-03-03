import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# 日本語フォント設定 (Windows向け)
plt.rcParams["font.family"] = "MS Gothic"


# ==========================================
# 1. 設定
# ==========================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
    INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
    OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_ridge_strict"

    SEED = 42
    TRAIN_END = "2023-12-31"
    VAL_END = "2024-12-31"

    # 探索設定
    N_TRIALS = 50

    # ターゲット生成用 (5日後予測)
    PRED_HORIZON = 5


Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. ユーティリティ
# ==========================================
def calculate_metrics_df(df_res):
    """深層学習版と共通の評価指標計算"""
    y_true_price = df_res["Actual"].values
    y_pred_price = df_res["Pred"].values
    y_curr_price = df_res["Current"].values

    # リターンベースの指標
    actual_return = (y_true_price - y_curr_price) / y_curr_price
    pred_return = df_res["Pred_Return"].values

    # 1. 方向正解率 (Accuracy)
    diff_true = y_true_price - y_curr_price
    diff_pred = y_pred_price - y_curr_price
    # 0変化を除外せずシンプルに符号比較
    accuracy = accuracy_score(np.sign(diff_true), np.sign(diff_pred)) * 100

    # 2. 決定係数 (R2)
    r2_price = r2_score(y_true_price, y_pred_price)
    r2_return = r2_score(actual_return, pred_return)

    # 3. 誤差
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae = mean_absolute_error(y_true_price, y_pred_price)

    # 4. 相関
    corr = np.corrcoef(y_true_price, y_pred_price)[0, 1] if len(y_true_price) > 1 else np.nan

    return pd.Series(
        {"RMSE": rmse, "MAE": mae, "Accuracy": accuracy, "R2_Price": r2_price, "R2_Return": r2_return, "Corr": corr}
    )


# ==========================================
# 3. データ読み込み & 前処理
# ==========================================
def load_data():
    print(f"[Loading] {Config.INPUT_FILE}")
    if not Config.INPUT_FILE.exists():
        print(f"[Error] File not found: {Config.INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(Config.INPUT_FILE, dtype={"Code": str}, low_memory=False)

    # カラム名統一 (Code -> code)
    if "Code" in df.columns:
        df.rename(columns={"Code": "code"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["code", "Date"], inplace=True)

    # --- ターゲット変数の生成 (5日後予測) ---
    # FusionTransformer等と条件を揃えるため、shift(-5)を使用
    target_col = "Target_Return"
    print(f"[Preprocessing] Generating {target_col} with shift(-{Config.PRED_HORIZON})...")
    df[target_col] = df.groupby("code")["Close"].transform(lambda x: np.log(x.shift(-Config.PRED_HORIZON) / x))

    # 欠損除去
    df = df.dropna(subset=[target_col]).fillna(0)

    # 特徴量の選定 (リークになりうる未来情報は除外)
    exclude_cols = [
        "Date",
        "code",
        "Name",
        "Sector",
        target_col,
        "Target_Close_5D",
        "Target_Return_1D",
        "Target_Return_5D",
    ]
    feature_cols = [c for c in df.columns if c not in exclude_cols and np.issubdtype(df[c].dtype, np.number)]

    print(f"Features: {len(feature_cols)}")

    # データ分割
    dates = df["Date"]
    train_mask = dates <= Config.TRAIN_END
    val_mask = (dates > Config.TRAIN_END) & (dates <= Config.VAL_END)
    test_mask = dates > Config.VAL_END

    # --- 厳密なスケーリング (Trainのみでfit) ---
    scaler = StandardScaler()
    X_train_raw = df.loc[train_mask, feature_cols].values
    scaler.fit(X_train_raw)

    # 全データを変換
    X_all = scaler.transform(df[feature_cols].values)
    y_all = df[target_col].values

    # データセット辞書作成
    data = {
        "train_X": X_all[train_mask],
        "train_y": y_all[train_mask],
        "val_X": X_all[val_mask],
        "val_y": y_all[val_mask],
        "test_X": X_all[test_mask],
        "test_y": y_all[test_mask],
        "features": feature_cols,
        # 分析用メタデータ (Test期間)
        "meta_test": df.loc[test_mask, ["Date", "code", "Name", "Close"]].reset_index(drop=True),
        "scaler": scaler,
    }

    print(f"Train: {train_mask.sum()}, Val: {val_mask.sum()}, Test: {test_mask.sum()}")
    return data


# ==========================================
# 4. Optuna Objective
# ==========================================
def objective(trial, data):
    # Ridgeのハイパーパラメータ探索
    alpha = trial.suggest_float("alpha", 0.01, 1000.0, log=True)

    model = Ridge(alpha=alpha, random_state=Config.SEED)
    model.fit(data["train_X"], data["train_y"])

    # ValidationデータでのMSEで評価
    preds = model.predict(data["val_X"])
    mse = mean_squared_error(data["val_y"], preds)
    return mse


# ==========================================
# 5. メイン実行
# ==========================================
def main():
    start_time = time.time()
    data = load_data()

    print("\n--- [Ridge] Hyperparameter Tuning ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, data), n_trials=Config.N_TRIALS)

    print(f"Best Params: {study.best_params}")

    # --- 厳密な最終評価 ---
    # Deep Learning版と同様、Trainデータで学習したモデルをTestで評価する
    # (Train+Valで再学習する手法もあるが、DL版との条件統一のためTrainのみで学習したベストモデルを採用する)

    best_model = Ridge(alpha=study.best_params["alpha"], random_state=Config.SEED)
    best_model.fit(data["train_X"], data["train_y"])

    # Testデータ予測
    print("\n--- Evaluating on Test Data ---")
    pred_return = best_model.predict(data["test_X"])

    # 結果集計
    df_res = data["meta_test"].copy()

    # 実測価格の復元 (Close * e^Target_Return) or (Close * (1+Target_Return))
    # ここでは対数収益率として計算したので exp で戻すのが厳密だが、
    # 簡易的に (1+r) 近似でも可。Deep版と合わせるため、データセット作成時の定義に従う。
    # ターゲット作成時: log(P_future / P_current) -> P_future = P_current * exp(Target)
    df_res["Current"] = df_res["Close"]
    df_res["Actual"] = df_res["Close"] * np.exp(data["test_y"])

    # 予測価格
    df_res["Pred_Return"] = pred_return
    df_res["Pred"] = df_res["Close"] * np.exp(pred_return)

    # 指標算出
    metrics = calculate_metrics_df(df_res)
    print("\n[Test Scores]")
    print(metrics)

    # 保存
    model_name = "Ridge"
    res_path = Config.OUTPUT_DIR / f"predictions_{model_name}.csv"
    df_res.to_csv(res_path, index=False)

    # 特徴量重要度 (係数の絶対値)
    importance = pd.DataFrame({"Feature": data["features"], "Importance": np.abs(best_model.coef_)}).sort_values(
        "Importance", ascending=False
    )
    importance.to_csv(Config.OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False)

    # 散布図プロット
    plt.figure(figsize=(6, 6))
    plt.scatter(data["test_y"], pred_return, alpha=0.3, s=10)
    plt.plot([data["test_y"].min(), data["test_y"].max()], [data["test_y"].min(), data["test_y"].max()], "r--")
    plt.title(f"{model_name}: Actual vs Predicted Return")
    plt.xlabel("Actual Return (Log)")
    plt.ylabel("Predicted Return (Log)")
    plt.grid(True, alpha=0.3)
    plt.savefig(Config.OUTPUT_DIR / f"{model_name}_scatter.png")
    plt.close()

    elapsed = time.time() - start_time
    print(f"\n[Done] Ridge Evaluation Completed ({elapsed:.1f}s)")
    print(f"Results saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
