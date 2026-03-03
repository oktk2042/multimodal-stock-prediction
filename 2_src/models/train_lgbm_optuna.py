import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

plt.rcParams["font.family"] = "MS Gothic"


# ==========================================
# 1. 設定
# ==========================================
class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
    INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
    OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_lgbm_strict"

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

    actual_return = (y_true_price - y_curr_price) / y_curr_price
    pred_return = df_res["Pred_Return"].values

    # Accuracy
    diff_true = y_true_price - y_curr_price
    diff_pred = y_pred_price - y_curr_price
    accuracy = accuracy_score(np.sign(diff_true), np.sign(diff_pred)) * 100

    # R2
    r2_price = r2_score(y_true_price, y_pred_price)
    r2_return = r2_score(actual_return, pred_return)

    # Error
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))
    mae = mean_absolute_error(y_true_price, y_pred_price)

    corr = np.corrcoef(y_true_price, y_pred_price)[0, 1] if len(y_true_price) > 1 else np.nan

    return pd.Series(
        {"RMSE": rmse, "MAE": mae, "Accuracy": accuracy, "R2_Price": r2_price, "R2_Return": r2_return, "Corr": corr}
    )


# ==========================================
# 3. データ読み込み
# ==========================================
def load_data():
    print(f"[Loading] {Config.INPUT_FILE}")
    if not Config.INPUT_FILE.exists():
        print(f"[Error] File not found: {Config.INPUT_FILE}")
        sys.exit(1)

    df = pd.read_csv(Config.INPUT_FILE, dtype={"Code": str}, low_memory=False)

    if "Code" in df.columns:
        df.rename(columns={"Code": "code"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["code", "Date"], inplace=True)

    # --- ターゲット生成 (5日後予測) ---
    target_col = "Target_Return"
    print(f"[Preprocessing] Generating {target_col} with shift(-{Config.PRED_HORIZON})...")
    df[target_col] = df.groupby("code")["Close"].transform(lambda x: np.log(x.shift(-Config.PRED_HORIZON) / x))

    df = df.dropna(subset=[target_col]).fillna(0)

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

    # 分割
    dates = df["Date"]
    train_df = df[dates <= Config.TRAIN_END]
    val_df = df[(dates > Config.TRAIN_END) & (dates <= Config.VAL_END)]
    test_df = df[dates > Config.VAL_END]

    # LGBM Dataset作成
    lgb_train = lgb.Dataset(train_df[feature_cols], train_df[target_col], free_raw_data=False)
    lgb_val = lgb.Dataset(val_df[feature_cols], val_df[target_col], reference=lgb_train, free_raw_data=False)

    # テスト用
    test_X = test_df[feature_cols]
    test_y = test_df[target_col].values

    # データセット辞書
    data = {
        "lgb_train": lgb_train,
        "lgb_val": lgb_val,
        "test_X": test_X,
        "test_y": test_y,
        "features": feature_cols,
        "meta_test": test_df[["Date", "code", "Name", "Close"]].reset_index(drop=True),
    }

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return data


# ==========================================
# 4. Optuna Objective
# ==========================================
def objective(trial, data):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": Config.SEED,
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    # Train dataを用いて学習し、Val dataでEarly Stopping
    model = lgb.train(
        params,
        data["lgb_train"],
        valid_sets=[data["lgb_train"], data["lgb_val"]],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(0),  # ログ抑制
        ],
    )

    preds = model.predict(data["lgb_val"].data)
    rmse = np.sqrt(mean_squared_error(data["lgb_val"].label, preds))
    return rmse


# ==========================================
# 5. メイン実行
# ==========================================
def main():
    start_time = time.time()
    data = load_data()

    print("\n--- [LightGBM] Hyperparameter Tuning ---")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: objective(t, data), n_trials=Config.N_TRIALS)

    print(f"Best Params: {study.best_params}")

    # --- 厳密な最終モデル学習 ---
    # ベストパラメータを用いて、Trainデータで学習 (Valを監視役にEarly Stopping)
    best_params = study.best_params
    best_params.update({"objective": "regression", "metric": "rmse", "verbosity": -1, "seed": Config.SEED})

    print("\n--- Training Best Model ---")
    best_model = lgb.train(
        best_params,
        data["lgb_train"],
        valid_sets=[data["lgb_train"], data["lgb_val"]],
        num_boost_round=2000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)],
    )

    # Testデータ予測
    print("\n--- Evaluating on Test Data ---")
    pred_return = best_model.predict(data["test_X"])

    # 結果集計
    df_res = data["meta_test"].copy()

    # 価格復元: P_future = P_current * exp(Target)
    df_res["Current"] = df_res["Close"]
    df_res["Actual"] = df_res["Close"] * np.exp(data["test_y"])

    df_res["Pred_Return"] = pred_return
    df_res["Pred"] = df_res["Close"] * np.exp(pred_return)

    metrics = calculate_metrics_df(df_res)
    print("\n[Test Scores]")
    print(metrics)

    # 保存
    model_name = "LightGBM"
    res_path = Config.OUTPUT_DIR / f"predictions_{model_name}.csv"
    df_res.to_csv(res_path, index=False)

    # 特徴量重要度
    importance = pd.DataFrame(
        {"Feature": data["features"], "Importance": best_model.feature_importance(importance_type="gain")}
    ).sort_values("Importance", ascending=False)
    importance.to_csv(Config.OUTPUT_DIR / f"{model_name}_feature_importance.csv", index=False)

    # 散布図
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
    print(f"\n[Done] LightGBM Evaluation Completed ({elapsed:.1f}s)")
    print(f"Results saved to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
