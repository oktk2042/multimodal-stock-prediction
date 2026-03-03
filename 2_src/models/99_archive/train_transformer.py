import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
import copy
import math

# 設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase2_transformer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# パラメータ
SEQ_LEN = 30
BATCH_SIZE = 64
D_MODEL = 64       # 特徴量次元 (埋め込み次元)
NHEAD = 4          # ヘッド数
NUM_LAYERS = 2     # Encoder層数
DROPOUT = 0.1
EPOCHS = 30
LEARNING_RATE = 0.0005 # Transformerは学習率小さめ推奨
TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len
    def __len__(self): return len(self.X) - self.seq_len
    def __getitem__(self, idx):
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len - 1]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout):
        super(TransformerModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, d_model) # 入力次元をd_modelに変換
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x)
        # 最後のタイムステップのみ使用 (Global Average Poolingもアリだが今回はLast Step)
        output = output[:, -1, :] 
        output = self.fc(output)
        return output.squeeze()

def train_model():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
    
    target_col = 'Target_Return_5D'
    price_col = 'Close'
    actual_price_col = 'Target_Close_5D'
    exclude = ['Date', 'code', 'Name', 'Sector', target_col, actual_price_col, 'Target_Return_1D', price_col]
    feature_cols = [c for c in df.columns if c not in exclude]
    
    scaler = StandardScaler()
    feature_data = scaler.fit_transform(df[feature_cols].fillna(0).values)
    target_data = df[target_col].fillna(0).values
    
    dates = df['Date']
    train_idx = dates[dates <= TRAIN_END].index
    val_idx   = dates[(dates > TRAIN_END) & (dates <= VAL_END)].index
    test_idx  = dates[dates > VAL_END].index
    
    train_loader = DataLoader(StockDataset(feature_data[train_idx], target_data[train_idx], SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(StockDataset(feature_data[val_idx], target_data[val_idx], SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(StockDataset(feature_data[test_idx], target_data[test_idx], SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False)
    
    model = TransformerModel(len(feature_cols), D_MODEL, NHEAD, NUM_LAYERS, DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses, val_losses = [], []
    
    print("Transformer Training Start...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train: {epoch_loss:.6f} | Val: {val_epoch_loss:.6f}")
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), OUTPUT_DIR / "best_transformer_model.pth")
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Transformer Learning Curve')
    plt.savefig(OUTPUT_DIR / 'transformer_learning_curve.png')
    
    # 評価
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs.to(DEVICE))
            preds.extend(outputs.cpu().numpy())
            actuals.extend(labels.numpy())
            
    preds = np.array(preds)
    valid_test_idx = test_idx[SEQ_LEN:]
    price_test = df.loc[valid_test_idx, price_col].values
    actual_price_test = df.loc[valid_test_idx, actual_price_col].values
    pred_price = price_test * (1 + preds)
    
    rmse = np.sqrt(mean_squared_error(actual_price_test, pred_price))
    diff_true = actual_price_test - price_test
    diff_pred = pred_price - price_test
    acc = accuracy_score(np.sign(diff_true), np.sign(diff_pred)) * 100
    
    print("\n🏆 Final Test Score (Transformer):")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  Accuracy: {acc:.2f}%")
    
    with open(OUTPUT_DIR / "transformer_result.txt", "w") as f:
        f.write(f"RMSE: {rmse}\nAccuracy: {acc}\n")

if __name__ == "__main__":
    train_model()