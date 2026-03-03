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

# 設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase2_lstm"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# パラメータ
SEQ_LEN = 30       # 過去30日分を見る
BATCH_SIZE = 64
HIDDEN_DIM = 64
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 30        # Early Stoppingあり
LEARNING_RATE = 0.001
TRAIN_END = "2023-12-31"
VAL_END   = "2024-12-31"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class StockDataset(Dataset):
    def __init__(self, X, y, seq_len=30):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        # idx から idx+seq_len までのデータを取得 (Sequence)
        x_seq = self.X[idx : idx + self.seq_len]
        # ターゲットは seq_len 番目の値 (Sequenceの次の日)
        # ※ dataset作成時に既にshiftされているので、idx + seq_len - 1 が対応する行
        # ただし今回のdatasetは「その行のTarget」が入っているので、
        # シーケンスの最後の行のTargetを取得すればよい
        y_label = self.y[idx + self.seq_len - 1] 
        return x_seq, y_label

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        # 最後のタイムステップの出力を使用
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze()

def train_model():
    print(f"Using device: {DEVICE}")
    
    # 1. データロード
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
    df['Date'] = pd.to_datetime(df['Date'])
    
    target_col = 'Target_Return_5D'
    price_col = 'Close'
    actual_price_col = 'Target_Close_5D'
    exclude_cols = ['Date', 'code', 'Name', 'Sector', target_col, actual_price_col, 'Target_Return_1D', price_col]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # 2. 前処理 (StandardScaler) - Deep Learningには必須
    scaler = StandardScaler()
    feature_data = df[feature_cols].fillna(0).values
    feature_data = scaler.fit_transform(feature_data)
    target_data = df[target_col].fillna(0).values
    
    # 3. データ分割 (インデックスベース)
    dates = df['Date']
    train_idx = dates[dates <= TRAIN_END].index
    val_idx   = dates[(dates > TRAIN_END) & (dates <= VAL_END)].index
    test_idx  = dates[dates > VAL_END].index
    
    # シーケンス用にDataset作成
    # インデックスを直接渡して、Dataset内でスライスする方式はメモリ効率が良いが
    # ここでは簡単のため、分割したデータを渡す
    
    # 注意: 時系列が連続している必要があるため、codeごとにグルーピングしてDatasetを作るのが本来だが、
    # 簡易的に全体を時系列ソートしてある前提で処理する。(Top200データは code, Date ソート済み)
    # 厳密には銘柄の変わり目で不連続になるが、データ数が多いため影響は軽微として許容する。
    
    train_dataset = StockDataset(feature_data[train_idx], target_data[train_idx], SEQ_LEN)
    val_dataset   = StockDataset(feature_data[val_idx], target_data[val_idx], SEQ_LEN)
    test_dataset  = StockDataset(feature_data[test_idx], target_data[test_idx], SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True) # TrainはシャッフルOK
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4. モデル構築
    model = LSTMModel(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, 
                      num_layers=NUM_LAYERS, dropout=DROPOUT).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 5. 学習ループ
    best_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    train_losses = []
    val_losses = []
    
    print("Training Start...")
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
        
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        
        val_epoch_loss = val_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_epoch_loss:.6f}")
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            
    # ベストモデルロード
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), OUTPUT_DIR / "best_lstm_model.pth")
    
    # 学習曲線
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('LSTM Learning Curve')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss')
    plt.legend(); plt.savefig(OUTPUT_DIR / 'lstm_learning_curve.png')
    
    # 6. テスト評価 & 指標計算
    model.eval()
    preds = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            preds.extend(outputs.cpu().numpy())
            actuals.extend(labels.numpy())
            
    preds = np.array(preds)
    actuals = np.array(actuals)
    
    # データセット作成時のSEQ_LEN分のズレを考慮して、価格データを取得
    # test_idx の先頭 SEQ_LEN 分は入力に使われるため、予測値が出るのは SEQ_LEN 番目から
    valid_test_idx = test_idx[SEQ_LEN:]
    
    price_test = df.loc[valid_test_idx, price_col].values
    actual_price_test = df.loc[valid_test_idx, actual_price_col].values
    pred_price = price_test * (1 + preds)
    
    # 指標計算
    rmse = np.sqrt(mean_squared_error(actual_price_test, pred_price))
    mae = mean_absolute_error(actual_price_test, pred_price)
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((actual_price_test - pred_price) / actual_price_test)) * 100
        mape = np.nan_to_num(mape)
    r2 = r2_score(actual_price_test, pred_price)
    
    diff_true = actual_price_test - price_test
    diff_pred = pred_price - price_test
    acc = accuracy_score(np.sign(diff_true), np.sign(diff_pred)) * 100
    mda = np.mean(np.sign(diff_true) == np.sign(diff_pred)) * 100
    
    print("\n🏆 Final Test Score (LSTM):")
    print(f"  RMSE: {rmse:,.2f}")
    print(f"  Accuracy: {acc:.2f}%")
    
    with open(OUTPUT_DIR / "lstm_result.txt", "w") as f:
        f.write(f"RMSE: {rmse}\nAccuracy: {acc}\nMDA: {mda}\n")

if __name__ == "__main__":
    train_model()