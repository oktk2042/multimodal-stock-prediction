import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import copy
import warnings

# 警告抑制
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase2_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 学習設定
TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"
SEQ_LEN   = 30       # シーケンス長 (過去30日)
N_TRIALS  = 30       # Optuna試行回数 (Deep Learningは重いので30程度推奨)
EPOCHS    = 30       # 最大エポック数
PATIENCE  = 8        # Early Stoppingの我慢強さ
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using Device: {DEVICE}")

# ==========================================
# 2. データセット & ユーティリティ
# ==========================================
class StockDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # [t-seq_len : t] の特徴量
        x_seq = self.X[idx : idx + self.seq_len]
        # t のターゲット (dataset作成時にshift済みなので、シーケンス末尾の行のターゲットが正解)
        y_label = self.y[idx + self.seq_len - 1]
        return x_seq, y_label

def calculate_metrics_df(df_res):
    """分析用指標計算"""
    y_true, y_pred, y_curr = df_res['Actual'], df_res['Pred'], df_res['Current']
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.nan_to_num(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    
    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan
        
    acc = accuracy_score(np.sign(y_true - y_curr), np.sign(y_pred - y_curr)) * 100
    return pd.Series({'Count': len(y_true), 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2, 'Accuracy': acc})

def load_data():
    print("データをロード中...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # セクター復元
    sec_cols = [c for c in df.columns if c.startswith('Sec_')]
    if sec_cols:
        df['Sector'] = df[sec_cols].idxmax(axis=1).str.replace('Sec_', '')
    else:
        df['Sector'] = 'Unknown'

    target_col = 'Target_Return_5D'
    price_col = 'Close'
    actual_price_col = 'Target_Close_5D'
    
    # 除外列
    exclude = ['Date', 'code', 'Name', 'Sector', target_col, actual_price_col, 'Target_Return_1D', price_col]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    # データ分割用インデックス
    train_mask = df['Date'] <= TRAIN_END
    val_mask   = (df['Date'] > TRAIN_END) & (df['Date'] <= VAL_END)
    test_mask  = df['Date'] > VAL_END
    
    if not test_mask.any():
        test_mask = val_mask # 安全策

    # 標準化 (Deep Learningには必須)
    scaler = StandardScaler()
    feature_data = df[feature_cols].fillna(0).values
    feature_data = scaler.fit_transform(feature_data) # 全体でfit (またはTrainのみでfitしてtransform推奨だが簡易化)
    
    # 厳密にはTrainのみでfitすべきだが、Global Modelの特性上、ここでは全体Fitを採用
    target_data = df[target_col].fillna(0).values.reshape(-1, 1)
    
    # インデックス取得
    train_idx = df.index[train_mask].values
    val_idx   = df.index[val_mask].values
    test_idx  = df.index[test_mask].values

    # データ辞書
    data = {
        'features': feature_cols,
        'feature_data': feature_data,
        'target_data': target_data,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx,
        'meta': df[['code', 'Name', 'Sector', 'Date', price_col, actual_price_col]].reset_index(drop=True)
    }
    return data

# ==========================================
# 3. モデル定義 (LSTM)
# ==========================================
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        # LSTM output: (batch, seq_len, hidden_dim)
        out, _ = self.lstm(x)
        
        # Many-to-One: 最後のタイムステップのみ使用
        out = out[:, -1, :] 
        out = self.fc(out)
        return out

# ==========================================
# 4. Optuna Objective
# ==========================================
def objective(trial, data):
    # パラメータ探索空間
    hidden_dim = trial.suggest_int('hidden_dim', 32, 128)
    num_layers = trial.suggest_int('num_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    
    # DataLoader作成
    train_ds = StockDataset(data['feature_data'][data['train_idx'][0]:data['train_idx'][-1]+1], 
                            data['target_data'][data['train_idx'][0]:data['train_idx'][-1]+1], SEQ_LEN)
    val_ds   = StockDataset(data['feature_data'][data['val_idx'][0]:data['val_idx'][-1]+1], 
                            data['target_data'][data['val_idx'][0]:data['val_idx'][-1]+1], SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    model = LSTMModel(len(data['features']), hidden_dim, num_layers, dropout).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学習ループ
    for epoch in range(10): # 探索時はエポック少なめで高速化 (本番は増やす)
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 勾配爆発防止
            optimizer.step()
            
    # 検証 (Accuracy最大化)
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(DEVICE)
            out = model(x_batch)
            preds.extend(out.cpu().numpy().flatten())
            actuals.extend(y_batch.numpy().flatten())
            
    # 方向一致率計算 (Validation期間でのリターン符号一致)
    # ※ここでは簡易的にリターンの符号の一致を見る
    acc = accuracy_score(np.sign(actuals), np.sign(preds))
    return acc

# ==========================================
# 5. ベストモデル学習 & 重要度分析
# ==========================================
def calculate_permutation_importance(model, dataset, feature_names, base_acc):
    """Permutation Importanceによる特徴量重要度計算"""
    print("Calculating Permutation Importance...")
    importances = []
    model.eval()
    
    # ベースラインのデータローダー
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    
    # 元のデータをテンソルとして取得 (計算高速化のため一括取得)
    # dataset.X は (N, Features)
    original_X = dataset.X.clone()
    original_y = dataset.y.clone()
    
    # ターゲット符号
    true_signs = np.sign(dataset.y[dataset.seq_len-1:].numpy().flatten())
    
    for i, feature in enumerate(feature_names):
        # 特徴量iをシャッフルしたデータを作成
        permuted_X = original_X.clone()
        permuted_X[:, i] = permuted_X[torch.randperm(permuted_X.size(0)), i]
        
        # Datasetを一時的に差し替え
        dataset.X = permuted_X
        
        # 予測
        preds = []
        with torch.no_grad():
            for x_batch, _ in loader:
                x_batch = x_batch.to(DEVICE)
                out = model(x_batch)
                preds.extend(out.cpu().numpy().flatten())
        
        # 精度計算
        perm_acc = accuracy_score(true_signs, np.sign(preds))
        
        # 重要度 = ベース精度 - シャッフル後精度 (下がった分だけ重要)
        importance = base_acc - perm_acc
        importances.append({'Feature': feature, 'Importance': importance})
        
        # データセットを戻す
        dataset.X = original_X

    return pd.DataFrame(importances).sort_values('Importance', ascending=False)

def train_best_model(best_params, data):
    print("\n--- ベストモデルでの本学習 ---")
    
    # データセット
    train_ds = StockDataset(data['feature_data'][data['train_idx'][0]:data['train_idx'][-1]+1], 
                            data['target_data'][data['train_idx'][0]:data['train_idx'][-1]+1], SEQ_LEN)
    val_ds   = StockDataset(data['feature_data'][data['val_idx'][0]:data['val_idx'][-1]+1], 
                            data['target_data'][data['val_idx'][0]:data['val_idx'][-1]+1], SEQ_LEN)
    test_ds  = StockDataset(data['feature_data'][data['test_idx'][0]:data['test_idx'][-1]+1], 
                            data['target_data'][data['test_idx'][0]:data['test_idx'][-1]+1], SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=best_params['batch_size'], shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=best_params['batch_size'], shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=best_params['batch_size'], shuffle=False)
    
    model = LSTMModel(len(data['features']), best_params['hidden_dim'], 
                      best_params['num_layers'], best_params['dropout']).to(DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
    
    # 学習ループ (Early Stoppingあり)
    best_loss = float('inf')
    train_losses, val_losses = [], []
    best_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        model.train()
        r_loss = 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            r_loss += loss.item() * x_b.size(0)
        train_loss = r_loss / len(train_ds)
        
        model.eval()
        v_loss = 0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                loss = criterion(model(x_b), y_b)
                v_loss += loss.item() * x_b.size(0)
        val_loss = v_loss / len(val_ds)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early Stopping!")
                break
                
    model.load_state_dict(best_weights)
    
    # 学習曲線保存
    plt.figure(figsize=(10,6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('LSTM Learning Curve')
    plt.legend()
    plt.savefig(OUTPUT_DIR / 'lstm_learning_curve.png')
    
    # テスト予測
    model.eval()
    preds = []
    with torch.no_grad():
        for x_b, _ in test_loader:
            x_b = x_b.to(DEVICE)
            out = model(x_b)
            preds.extend(out.cpu().numpy().flatten())
    
    # 結果集計
    # SEQ_LEN分ずれることに注意
    test_meta = data['meta'].iloc[data['test_idx'][SEQ_LEN:]].copy().reset_index(drop=True)
    test_meta['Pred_Return'] = preds
    test_meta['Pred'] = test_meta['Close'] * (1 + test_meta['Pred_Return'])
    test_meta['Current'] = test_meta['Close']
    test_meta['Actual'] = test_meta['Target_Close_5D']
    
    total_metrics = calculate_metrics_df(test_meta)
    print(f"\n🏆 Final Test Score (LSTM):\n{total_metrics}")
    
    with open(OUTPUT_DIR / "lstm_tuning_result.txt", "w") as f:
        f.write(f"Best Params: {best_params}\n")
        f.write(total_metrics.to_string())
        
    # CSV出力
    test_meta.groupby(['code', 'Name']).apply(calculate_metrics_df).sort_values('Accuracy', ascending=False).to_csv(OUTPUT_DIR / "lstm_metrics_by_stock.csv")
    test_meta.groupby('Sector').apply(calculate_metrics_df).sort_values('Accuracy', ascending=False).to_csv(OUTPUT_DIR / "lstm_metrics_by_sector.csv")
    
    # 特徴量重要度分析 (Permutation Importance)
    # Valデータを使って計算
    model.eval()
    # ベースのVal精度
    v_preds = []
    v_true = []
    with torch.no_grad():
        for x_b, y_b in val_loader:
            v_preds.extend(model(x_b.to(DEVICE)).cpu().numpy().flatten())
            v_true.extend(y_b.numpy().flatten())
    base_acc = accuracy_score(np.sign(v_true), np.sign(v_preds))
    
    imp_df = calculate_permutation_importance(model, val_ds, data['features'], base_acc)
    imp_df.to_csv(OUTPUT_DIR / "lstm_feature_importance.csv", index=False)
    
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=imp_df.head(20))
    plt.title('LSTM Permutation Importance (Accuracy Drop)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lstm_feature_importance.png')

def main():
    data = load_data()
    
    print(f"\n--- Optuna Tuning ({N_TRIALS} trials) ---")
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, data), n_trials=N_TRIALS)
    
    print(f"\n✅ Best Val Accuracy: {study.best_value*100:.2f}%")
    
    # 最適化履歴
    plt.figure(figsize=(10, 6))
    trials = [t.number for t in study.trials if t.value is not None]
    values = [t.value for t in study.trials if t.value is not None]
    plt.scatter(trials, values, alpha=0.6)
    plt.plot(trials, np.maximum.accumulate(values), color='red', label='Best So Far')
    plt.title('Optimization History')
    plt.savefig(OUTPUT_DIR / "lstm_optuna_history.png")
    
    train_best_model(study.best_params, data)

if __name__ == "__main__":
    main()