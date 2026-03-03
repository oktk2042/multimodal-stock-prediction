import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
import math
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time

# --- 0. 設定 (Configuration) ---
class Config:
    # --- ファイルパス ---
    BASE_PATH = Path("C:/M2_Research_Project/1_data")
    MODELING_PATH = BASE_PATH / "modeling_data"
    MODEL_SAVE_PATH = MODELING_PATH / "best_model_v2.pth"
    SCALER_PATH = MODELING_PATH / "feature_scaler_v2.gz"
    
    # --- データセットパラメータ ---
    TRAIN_DATA_PATH = MODELING_PATH / "train_data_v2.npz"
    VALID_DATA_PATH = MODELING_PATH / "validation_data_v2.npz"
    TEST_DATA_PATH = MODELING_PATH / "test_data_v2.npz"

    # --- モデルのハイパーパラメータ ---
    SEQUENCE_LENGTH = 30     # 入力として使用する過去の日数
    TARGET_OFFSET = 1        # 何日後を予測するか (1は翌日)
    INPUT_FEATURES = 10      # 使用する特徴量の数 (prepare_data.pyの出力に合わせる)
    D_MODEL = 128            # モデルの内部次元 (埋め込み次元)
    N_HEADS = 8              # Multi-Head Attentionのヘッド数
    N_ENCODER_LAYERS = 3     # Transformer Encoder層の数
    DROPOUT = 0.1            # ドロップアウト率

    # --- 学習のハイパーパラメータ ---
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    PATIENCE = 5             # 早期終了のための我慢回数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. データセットの準備 (Dataset & DataLoader) ---
class StockDataset(Dataset):
    """
    銘柄ごとに区切られた時系列データから、スライドウィンドウ方式で
    入力系列とターゲットを生成するカスタムデータセット。
    """
    def __init__(self, data_path, seq_len, target_offset):
        data = np.load(data_path, allow_pickle=True)
        self.features = data['features']
        self.targets = data['target']
        self.codes = data['codes']
        
        self.seq_len = seq_len
        self.target_offset = target_offset
        
        # 銘柄コードごとにデータのインデックス範囲を計算
        self.indices = []
        unique_codes, counts = np.unique(self.codes, return_counts=True)
        current_pos = 0
        for count in counts:
            num_samples = count - self.seq_len - self.target_offset + 1
            if num_samples > 0:
                # この銘柄から生成できるサンプル=[開始インデックス, 終了インデックス]
                self.indices.append((current_pos, current_pos + num_samples))
            current_pos += count

        # 各サンプルのインデックスを事前計算
        self.samples = []
        for start_pos, end_pos in self.indices:
            # start_pos: 銘柄の最初のデータのインデックス
            # end_pos: この銘柄で作れる最後のサンプルのインデックス
            # 例: データが100日分、seq_len=30なら、70個のサンプルが作れる
            num_brand_samples = (end_pos - start_pos)
            for i in range(num_brand_samples):
                self.samples.append(start_pos + i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        end_idx = start_idx + self.seq_len
        target_idx = end_idx + self.target_offset - 1

        # float32に変換
        inputs = torch.tensor(self.features[start_idx:end_idx], dtype=torch.float32)
        target = torch.tensor(self.targets[target_idx], dtype=torch.float32)

        return inputs, target.unsqueeze(-1) # ターゲットの次元を [1] にする

# --- 2. モデルの定義 (Transformer) ---
class PositionalEncoding(nn.Module):
    """位置エンコーディング"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """時系列予測のためのTransformerモデル"""
    def __init__(self, n_features, d_model, n_heads, n_layers, dropout):
        super(TimeSeriesTransformer, self).__init__()
        
        self.input_embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, src):
        # src shape: (batch_size, seq_len, n_features)
        src = self.input_embedding(src) * math.sqrt(Config.D_MODEL)
        src = self.pos_encoder(src)
        
        # TransformerEncoderへの入力
        output = self.transformer_encoder(src)
        
        # 系列の最後の時点の出力を使用して予測
        output = output[:, -1, :]
        output = self.output_layer(output)
        return output

# --- 3. 学習と評価の実行関数 ---
def train_and_evaluate():
    print(f"--- デバイス: {Config.DEVICE} を使用します ---")
    
    # --- データローダーの準備 ---
    train_dataset = StockDataset(Config.TRAIN_DATA_PATH, Config.SEQUENCE_LENGTH, Config.TARGET_OFFSET)
    valid_dataset = StockDataset(Config.VALID_DATA_PATH, Config.SEQUENCE_LENGTH, Config.TARGET_OFFSET)
    test_dataset = StockDataset(Config.TEST_DATA_PATH, Config.SEQUENCE_LENGTH, Config.TARGET_OFFSET)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0)
    print("--- データセットの準備が完了しました ---")

    # --- モデル、損失関数、最適化手法の定義 ---
    model = TimeSeriesTransformer(
        n_features=Config.INPUT_FEATURES,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_ENCODER_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = nn.MSELoss() # 平均二乗誤差
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"--- モデルの準備が完了しました (パラメータ数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M) ---")

    # --- 学習ループ ---
    best_valid_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n--- モデルの学習を開始します ---")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        # 訓練
        model.train()
        total_train_loss = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        
        # 検証
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                total_valid_loss += loss.item()
        
        avg_valid_loss = total_valid_loss / len(valid_loader)
        
        elapsed_time = time.time() - start_time
        
        print(f"エポック {epoch+1}/{Config.EPOCHS} | "
              f"訓練損失: {avg_train_loss:.4f} | "
              f"検証損失: {avg_valid_loss:.4f} | "
              f"時間: {elapsed_time:.2f}秒")
        
        # 早期終了とモデル保存
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            epochs_no_improve = 0
            print(f"  -> 検証損失が改善しました。モデルを保存します: {Config.MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.PATIENCE:
                print(f"\n--- {Config.PATIENCE}エポック連続で検証損失が改善しなかったため、学習を早期終了します。---")
                break
    
    # --- テストデータによる最終評価 ---
    print("\n--- テストデータで最終評価を行います ---")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.eval()
    
    all_targets = []
    all_predictions = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(Config.DEVICE), targets.to(Config.DEVICE)
            outputs = model(inputs)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(outputs.cpu().numpy())

    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)

    print("\n--- ★★★ 最終評価結果 ★★★ ---")
    print(f"  Mean Squared Error (MSE): {mse:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Absolute Error (MAE): {mae:.4f}")
    print("---------------------------------")

if __name__ == '__main__':
    train_and_evaluate()
