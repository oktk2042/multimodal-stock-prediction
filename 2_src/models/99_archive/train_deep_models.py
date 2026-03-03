import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings
import sys

# 各モデルのインポート
from arch.transformer import VanillaTransformer
from arch.dlinear import DLinear
from arch.patchtst import PatchTST
from arch.itransformer import iTransformer

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase2_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"
SEQ_LEN   = 30
N_TRIALS  = 20 # 各モデル20回探索
EPOCHS    = 20
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# ユーティリティ
# ==========================================
class StockDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx): return self.X[idx:idx+self.seq_len], self.y[idx+self.seq_len-1]

def load_data():
    print("Loading Data...")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
    
    target_col = 'Target_Return_5D'
    exclude = ['Date', 'code', 'Name', 'Sector', target_col, 'Target_Close_5D', 'Target_Return_1D', 'Close']
    feature_cols = [c for c in df.columns if c not in exclude]
    
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(df[feature_cols].fillna(0).values)
    target_data = df[target_col].fillna(0).values
    
    dates = df['Date']
    train_idx = df.index[dates <= TRAIN_END].values
    val_idx   = df.index[(dates > TRAIN_END) & (dates <= VAL_END)].values
    test_idx  = df.index[dates > VAL_END].values
    
    return {
        'feature_data': feature_data, 'target_data': target_data,
        'train_idx': train_idx, 'val_idx': val_idx, 'test_idx': test_idx,
        'input_dim': len(feature_cols)
    }

# ==========================================
# 学習関数
# ==========================================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)

def evaluate(model, loader):
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            preds.extend(out.cpu().numpy().flatten())
            actuals.extend(y.cpu().numpy().flatten())
    
    # Accuracy (方向一致率)
    acc = accuracy_score(np.sign(actuals), np.sign(preds))
    return acc

# ==========================================
# Optuna Objective
# ==========================================
def objective(trial, data, model_name):
    input_dim = data['input_dim']
    
    # 共通パラメータ
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    
    # モデル別パラメータ
    if model_name == "Transformer":
        model = VanillaTransformer(input_dim, SEQ_LEN, 1, 
                                   d_model=trial.suggest_categorical('d_model', [32, 64]),
                                   nhead=4, num_layers=trial.suggest_int('layers', 1, 3))
    elif model_name == "DLinear":
        model = DLinear(input_dim, SEQ_LEN, 1, individual=trial.suggest_categorical('individual', [True, False]))
    elif model_name == "PatchTST":
        model = PatchTST(input_dim, SEQ_LEN, 1, patch_len=10, stride=5,
                         d_model=trial.suggest_categorical('d_model', [32, 64]))
    elif model_name == "iTransformer":
        model = iTransformer(input_dim, SEQ_LEN, 1, d_model=trial.suggest_categorical('d_model', [32, 64]))
    
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # データローダー
    train_ds = StockDataset(data['feature_data'][data['train_idx'][0]:data['train_idx'][-1]+1], 
                            data['target_data'][data['train_idx'][0]:data['train_idx'][-1]+1], SEQ_LEN)
    val_ds   = StockDataset(data['feature_data'][data['val_idx'][0]:data['val_idx'][-1]+1], 
                            data['target_data'][data['val_idx'][0]:data['val_idx'][-1]+1], SEQ_LEN)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # 学習ループ
    best_acc = 0
    for epoch in range(5): # 探索時は高速化のため5エポック
        train_epoch(model, train_loader, optimizer, criterion)
        acc = evaluate(model, val_loader)
        best_acc = max(best_acc, acc)
        
    return best_acc

# ==========================================
# メイン実行
# ==========================================
def main():
    data = load_data()
    models = ["DLinear", "Transformer", "PatchTST", "iTransformer"]
    results = []
    
    for model_name in models:
        print(f"\n=== Tuning {model_name} ===")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, data, model_name), n_trials=N_TRIALS)
        
        print(f"Best Accuracy: {study.best_value:.4f}")
        print(f"Params: {study.best_params}")
        
        results.append({
            "Model": model_name,
            "Best_Val_Accuracy": study.best_value,
            "Best_Params": study.best_params
        })
        
    # 結果保存
    res_df = pd.DataFrame(results)
    res_df.to_csv(OUTPUT_DIR / "model_comparison_results.csv", index=False)
    print("\nAll Done! Results saved.")
    print(res_df)

if __name__ == "__main__":
    main()