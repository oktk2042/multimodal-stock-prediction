import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import lightgbm as lgb
import warnings

# 警告の抑制
warnings.filterwarnings('ignore')

# ==========================================
# 1. 設定 & 定数定義
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase1_baseline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 期間設定
TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"

# ==========================================
# 2. 評価指標計算関数 (MDA & Accuracy版)
# ==========================================
def calculate_metrics(y_true, y_pred, y_current=None):
    """
    株価予測に特化した評価指標を計算する
    """
    # 1. RMSE (円)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 2. MAE (円)
    mae = mean_absolute_error(y_true, y_pred)
    
    # 3. MAPE (%)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # 4. R2 Score
    r2 = r2_score(y_true, y_pred)
    
    # 方向予測の準備
    if y_current is not None:
        # 実際のリターンと予測リターン (または価格差)
        diff_true = y_true - y_current
        diff_pred = y_pred - y_current
        
        # 符号 (+1, -1, 0)
        sign_true = np.sign(diff_true)
        sign_pred = np.sign(diff_pred)
        
        # 5. Accuracy (単純一致率)
        accuracy = accuracy_score(sign_true, sign_pred) * 100
        
        # 6. MDA (Mean Directional Accuracy)
        # 定義: sign(y_true - y_current) == sign(y_pred - y_current) の平均
        # 実質的にAccuracyと同じ計算になることが多いですが、論文記載用に別変数として保持
        mda = np.mean(sign_true == sign_pred) * 100
    else:
        accuracy = np.nan
        mda = np.nan

    return rmse, mae, mape, r2, accuracy, mda

def evaluate_and_log(y_true, y_pred, y_current, model_name):
    rmse, mae, mape, r2, accuracy, mda = calculate_metrics(y_true, y_pred, y_current)
    
    print(f"\n📊 [{model_name}] 評価結果")
    print(f"  - RMSE (誤差): {rmse:,.2f} 円")
    print(f"  - MAE  (誤差): {mae:,.2f} 円")
    print(f"  - MAPE (誤差): {mape:.3f} %")
    print(f"  - R2   (説明): {r2:.4f}")
    print(f"  - Accuracy   : {accuracy:.2f} %")
    print(f"  - MDA        : {mda:.2f} %")
    
    return {
        "Model": model_name, 
        "RMSE": rmse, 
        "MAE": mae, 
        "MAPE": mape, 
        "R2": r2, 
        "Accuracy": accuracy, 
        "MDA": mda
    }

def train_baseline():
    print("--- Phase 1: Baseline Model Training (Metrics V4: MDA added) ---")
    
    if not INPUT_FILE.exists():
        print(f"エラー: {INPUT_FILE} が見つかりません。")
        return

    print("データをロード中...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
    
    target_col = 'Target_Return_5D'
    price_col = 'Close'
    actual_price_col = 'Target_Close_5D'
    
    exclude_cols = ['Date', 'code', 'Name', 'Target_Return_5D', 'Target_Close_5D', 'Target_Return_1D', 'Close']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"使用特徴量数: {len(feature_cols)}")

    # データ分割
    train_mask = df['Date'] <= TRAIN_END
    val_mask   = (df['Date'] > TRAIN_END) & (df['Date'] <= VAL_END)
    test_mask  = df['Date'] > VAL_END
    
    X_train = df.loc[train_mask, feature_cols]
    y_train = df.loc[train_mask, target_col]
    
    X_val   = df.loc[val_mask, feature_cols]
    y_val   = df.loc[val_mask, target_col]
    
    X_test  = df.loc[test_mask, feature_cols]
    
    price_test  = df.loc[test_mask, price_col]
    actual_test = df.loc[test_mask, actual_price_col]

    if len(X_test) == 0:
        print("⚠️ テストデータがありません。")
        return

    metrics_list = []

    # Model 1: Ridge
    print("\n[Model 1] Ridge Regression 学習中...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(0))
    X_test_scaled  = scaler.transform(X_test.fillna(0))
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_scaled, y_train)
    
    pred_return_ridge = ridge.predict(X_test_scaled)
    pred_price_ridge = price_test * (1 + pred_return_ridge)
    
    metrics_list.append(evaluate_and_log(actual_test, pred_price_ridge, price_test, "Ridge"))

    # Model 2: LightGBM
    print("\n[Model 2] LightGBM 学習中...")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val   = lgb.Dataset(X_val, y_val, reference=lgb_train)
    
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 31,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': 42
    }
    
    model = lgb.train(
        params,
        lgb_train,
        valid_sets=[lgb_train, lgb_val],
        num_boost_round=1000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=50),
            lgb.log_evaluation(0)
        ]
    )
    
    pred_return_lgb = model.predict(X_test, num_iteration=model.best_iteration)
    pred_price_lgb  = price_test * (1 + pred_return_lgb)
    
    metrics_list.append(evaluate_and_log(actual_test, pred_price_lgb, price_test, "LightGBM"))

    # 保存
    df_metrics = pd.DataFrame(metrics_list)
    print("\n--- 最終評価サマリー ---")
    print(df_metrics)
    df_metrics.to_csv(OUTPUT_DIR / "baseline_metrics_summary.csv", index=False)
    
    importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': model.feature_importance(importance_type='gain')
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance.head(20))
    plt.title('LightGBM Feature Importance (Top 20)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lgbm_feature_importance_v4.png')
    print(f"\n重要度グラフ保存: {OUTPUT_DIR / 'lgbm_feature_importance_v4.png'}")
    
    print("\n✅ Phase 1 完了")

if __name__ == "__main__":
    train_baseline()