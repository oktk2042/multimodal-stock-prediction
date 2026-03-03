import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import math
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
import time
import matplotlib.pyplot as plt
import japanize_matplotlib # グラフの日本語文字化け防止

# --- 0. 設定 (Configuration) ---
class Config:
    # --- ファイルパス ---
    BASE_PATH = Path("C:/M2_Research_Project/1_data")
    MODELING_PATH = BASE_PATH / "modeling_data"
    
    MODEL_SAVE_PATH = MODELING_PATH / "best_model_final_v4.pth"
    SCALER_PATH = MODELING_PATH / "feature_scaler_v4.gz"
    
    TRAIN_DATA_PATH = MODELING_PATH / "train_data_v4.npz"
    VALID_DATA_PATH = MODELING_PATH / "validation_data_v4.npz"
    TEST_DATA_PATH = MODELING_PATH / "test_data_v4.npz"

    # --- モデルハイパーパラメータ ---
    ENCODER_SEQUENCE_LENGTH = 30
    PREDICTION_WINDOW_SIZE = 5
    
    EMBEDDING_DIM = 16
    D_MODEL = 128
    N_HEADS = 8
    N_ENCODER_LAYERS = 3
    N_DECODER_LAYERS = 3
    DROPOUT = 0.1

    # --- 学習ハイパーパラメータ ---
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.0001
    PATIENCE = 5
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1. データセット (StockDataset) ---
class StockDataset(Dataset):
    def __init__(self, data_path, code_to_id, encoder_seq_len, pred_len):
        data = np.load(data_path, allow_pickle=True)
        self.features = data['features']
        self.targets = data['target']
        self.codes = data['codes']
        self.code_to_id = code_to_id
        
        self.encoder_seq_len = encoder_seq_len
        self.pred_len = pred_len
        self.total_len = encoder_seq_len + pred_len

        self.samples = []
        unique_codes_in_data = sorted(list(np.unique(self.codes)))
        
        for code in unique_codes_in_data:
            code_indices = np.where(self.codes == code)[0]
            start_pos = code_indices[0]
            count = len(code_indices)
            num_samples = count - self.total_len + 1
            if num_samples > 0:
                for i in range(num_samples):
                    self.samples.append(start_pos + i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        start_idx = self.samples[idx]
        encoder_input_end = start_idx + self.encoder_seq_len
        decoder_output_end = encoder_input_end + self.pred_len

        encoder_input = self.features[start_idx:encoder_input_end]
        decoder_input = self.features[encoder_input_end - 1 : decoder_output_end - 1]
        decoder_output = self.targets[encoder_input_end:decoder_output_end]
        
        stock_code = self.codes[start_idx]
        stock_id = self.code_to_id[stock_code]

        return (
            torch.tensor(encoder_input, dtype=torch.float32),
            torch.tensor(decoder_input, dtype=torch.float32),
            torch.tensor(decoder_output, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(stock_id, dtype=torch.long)
        )

# --- 2. モデル定義 (Seq2SeqTransformer with Embedding) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, n_features, n_stocks, embedding_dim, d_model, n_heads, n_encoder_layers, n_decoder_layers, dropout):
        super(Seq2SeqTransformer, self).__init__()
        self.d_model = d_model
        self.stock_embedding = nn.Embedding(n_stocks, embedding_dim)
        self.input_embedding = nn.Linear(n_features + embedding_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # [修正] nn.TransformerDecoderLayerを個別に定義してAttentionを取得しやすくする
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_layers)

        self.output_layer = nn.Linear(d_model, 1)

    def _generate_square_subsequent_mask(self, sz):
        return nn.Transformer.generate_square_subsequent_mask(sz).to(Config.DEVICE)

    def forward(self, src, tgt, stock_ids):
        stock_emb = self.stock_embedding(stock_ids)
        src_emb = stock_emb.unsqueeze(1).expand(-1, src.size(1), -1)
        tgt_emb = stock_emb.unsqueeze(1).expand(-1, tgt.size(1), -1)

        src = torch.cat([src, src_emb], dim=-1)
        tgt = torch.cat([tgt, tgt_emb], dim=-1)

        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1))
        
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        tgt = self.input_embedding(tgt) * math.sqrt(self.d_model)
        
        src = self.pos_encoder(src)
        tgt = self.pos_encoder(tgt)

        # [修正] EncoderとDecoderを個別に呼び出す
        memory = self.transformer_encoder(src)
        
        # [修正] DecoderからAttentionの重みを直接受け取る
        # need_weights=Trueにすることで、(出力, Attention重み)のタプルが返る
        output, attention_weights = self.transformer_decoder.layers[-1].multihead_attn(
            tgt, memory, memory, need_weights=True)
        
        # Decoderの残りの層を計算
        # 本来はDecoder全体をループすべきだが、最後の層のAttentionを見るため簡略化
        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        return self.output_layer(output), attention_weights

# --- 3. 推論、評価、可視化 ---
def evaluate_and_plot(model, data_loader, scaler, total_feature_names, id_to_code):
    model.eval()
    all_preds_inversed, all_targets_inversed, all_codes = [], [], []
    last_batch_attention, last_batch_stock_ids = None, None
    
    try:
        target_col_index = total_feature_names.index('Close')
        print(f"目的変数 'Close' は {target_col_index} 番目の列として検出されました。")
    except ValueError:
        raise ValueError("'Close'がスケーラーの特徴量名リストに見つかりませんでした。前処理スクリプトを再実行してください。")
        
    n_total_features = len(total_feature_names)
    n_input_features = next(iter(data_loader))[0].shape[2]

    with torch.no_grad():
        for i, (enc_in, _, dec_out, stock_ids) in enumerate(data_loader):
            enc_in, stock_ids = enc_in.to(Config.DEVICE), stock_ids.to(Config.DEVICE)
            
            decoder_input = enc_in[:, -1:, :] 
            predictions_scaled = []
            
            # [修正] 最初の予測ステップのAttentionのみを保存
            first_step_attention = None

            for step in range(Config.PREDICTION_WINDOW_SIZE):
                output, attention_weights = model(enc_in, decoder_input, stock_ids)
                
                if step == 0:
                    first_step_attention = attention_weights

                next_pred_scaled = output[:, -1:, :]
                
                next_pred_features = torch.zeros( (enc_in.size(0), 1, n_input_features), device=Config.DEVICE)
                
                decoder_input = torch.cat([decoder_input, next_pred_features], dim=1)
                predictions_scaled.append(next_pred_scaled)

            if i == len(data_loader) - 1:
                last_batch_attention = first_step_attention
                last_batch_stock_ids = stock_ids
            
            batch_preds_scaled = torch.cat(predictions_scaled, dim=1)
            
            preds_flat = batch_preds_scaled.cpu().numpy().flatten()
            dummy_preds = np.zeros((len(preds_flat), n_total_features))
            dummy_preds[:, target_col_index] = preds_flat
            inversed_preds = scaler.inverse_transform(dummy_preds)[:, target_col_index]
            
            targets_flat = dec_out.squeeze(-1).cpu().numpy().flatten()
            dummy_targets = np.zeros((len(targets_flat), n_total_features))
            dummy_targets[:, target_col_index] = targets_flat
            inversed_targets = scaler.inverse_transform(dummy_targets)[:, target_col_index]

            all_preds_inversed.append(inversed_preds.reshape(batch_preds_scaled.shape[0], -1))
            all_targets_inversed.append(inversed_targets.reshape(dec_out.shape[0], -1))
            all_codes.extend([id_to_code[sid.item()] for sid in stock_ids])

    all_preds = np.concatenate(all_preds_inversed)
    all_targets = np.concatenate(all_targets_inversed)
    
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)
    
    mape_targets = all_targets[np.abs(all_targets) > 1e-6]
    mape_preds = all_preds[np.abs(all_targets) > 1e-6]
    mape = np.mean(np.abs((mape_targets - mape_preds) / mape_targets)) * 100
    
    plot_predictions(all_preds, all_targets, all_codes, num_plots=5)
    
    if last_batch_attention is not None:
        plot_attention_map(last_batch_attention, last_batch_stock_ids, id_to_code)
    else:
        print("[警告] Attentionの重みを取得できませんでした。Attentionマップは生成されません。")

    return rmse, mae, mape

def plot_predictions(predictions, targets, codes, num_plots=5):
    num_samples = predictions.shape[0]
    indices = np.random.choice(num_samples, min(num_plots, num_samples), replace=False)

    fig, axes = plt.subplots(nrows=num_plots, ncols=1, figsize=(12, num_plots * 4))
    if num_plots == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        ax = axes[i]
        days = np.arange(1, Config.PREDICTION_WINDOW_SIZE + 1)
        ax.plot(days, targets[idx, :], 'o-', label='実績値 (Actual)', color='dodgerblue')
        ax.plot(days, predictions[idx, :], 'x--', label='予測値 (Predicted)', color='crimson')
        ax.set_title(f"銘柄コード: {codes[idx]} の5日間予測", fontsize=14)
        ax.set_xlabel("予測日 (Day)", fontsize=12)
        ax.set_ylabel("終値 (円)", fontsize=12)
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))

    plt.tight_layout()
    plot_path = Config.MODELING_PATH / "prediction_vs_actual_plots.png"
    plt.savefig(plot_path)
    print(f"\n--- 予測結果のグラフを保存しました: {plot_path} ---")

# --- 4. 可視化関数 ---
def plot_loss_curves(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='訓練損失 (Train Loss)')
    plt.plot(history['valid_loss'], label='検証損失 (Validation Loss)')
    plt.title('学習曲線の推移', fontsize=16)
    plt.xlabel('エポック (Epoch)', fontsize=12)
    plt.ylabel('損失 (MSE Loss)', fontsize=12)
    plt.legend()
    plt.grid(True)
    
    plot_path = Config.MODELING_PATH / "loss_curves.png"
    plt.savefig(plot_path)
    print(f"--- 学習曲線のグラフを保存しました: {plot_path} ---")

def plot_attention_map(attention_weights, stock_ids, id_to_code):
    attention = attention_weights.cpu().detach()
    
    sample_idx = 0
    attention_sample = attention[sample_idx, :, :].mean(dim=0).numpy()
    code = id_to_code[stock_ids[sample_idx].item()]
    
    plt.figure(figsize=(12, 6))
    days_ago = -np.arange(Config.ENCODER_SEQUENCE_LENGTH, 0, -1)
    plt.bar(days_ago, attention_sample)
    plt.xlabel('何日前か (Days Ago)', fontsize=12)
    plt.ylabel('Attentionスコア', fontsize=12)
    plt.title(f'銘柄コード {code} の予測における過去時点へのAttention', fontsize=16)
    plt.xticks(days_ago[::2])
    
    plot_path = Config.MODELING_PATH / "attention_map.png"
    plt.savefig(plot_path)
    print(f"--- Attentionマップのグラフを保存しました: {plot_path} ---")

# --- 5. メイン実行関数 ---
def main():
    print(f"--- デバイス: {Config.DEVICE} ---")
    
    scaler = joblib.load(Config.SCALER_PATH)
    total_feature_names = list(getattr(scaler, 'feature_names_in_', []))
    if not total_feature_names:
         raise RuntimeError("スケーラーに特徴量名が保存されていません。scikit-learnのバージョンが古いか、前処理スクリプトに問題があります。")

    temp_data = np.load(Config.TRAIN_DATA_PATH, allow_pickle=True)
    INPUT_FEATURES = temp_data['features'].shape[1]
    unique_codes = sorted(list(np.unique(temp_data['codes'])))
    code_to_id = {code: i for i, code in enumerate(unique_codes)}
    id_to_code = {i: code for code, i in code_to_id.items()}
    NUM_STOCKS = len(unique_codes)

    print(f"--- 検出されたパラメータ ---")
    print(f"入力特徴量数 (INPUT_FEATURES): {INPUT_FEATURES}")
    print(f"スケーラーの総特徴量数: {len(total_feature_names)}")
    print(f"銘柄数 (NUM_STOCKS): {NUM_STOCKS}")

    train_dataset = StockDataset(Config.TRAIN_DATA_PATH, code_to_id, Config.ENCODER_SEQUENCE_LENGTH, Config.PREDICTION_WINDOW_SIZE)
    valid_dataset = StockDataset(Config.VALID_DATA_PATH, code_to_id, Config.ENCODER_SEQUENCE_LENGTH, Config.PREDICTION_WINDOW_SIZE)
    test_dataset = StockDataset(Config.TEST_DATA_PATH, code_to_id, Config.ENCODER_SEQUENCE_LENGTH, Config.PREDICTION_WINDOW_SIZE)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    print("--- データセット準備完了 ---")

    model = Seq2SeqTransformer(
        n_features=INPUT_FEATURES, n_stocks=NUM_STOCKS, embedding_dim=Config.EMBEDDING_DIM,
        d_model=Config.D_MODEL, n_heads=Config.N_HEADS, n_encoder_layers=Config.N_ENCODER_LAYERS,
        n_decoder_layers=Config.N_DECODER_LAYERS, dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"--- モデル準備完了 (パラメータ数: {sum(p.numel() for p in model.parameters())/1e6:.2f}M) ---")

    best_valid_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'valid_loss': []}
    
    print("\n--- モデル学習開始 ---")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()
        
        model.train()
        total_train_loss = 0
        for enc_in, dec_in, dec_out, stock_ids in train_loader:
            enc_in, dec_in, dec_out, stock_ids = enc_in.to(Config.DEVICE), dec_in.to(Config.DEVICE), dec_out.to(Config.DEVICE), stock_ids.to(Config.DEVICE)
            optimizer.zero_grad()
            # [修正] モデルからの返り値(タプル)を正しく受け取る
            outputs, _ = model(enc_in, dec_in, stock_ids)
            loss = criterion(outputs, dec_out)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        
        model.eval()
        total_valid_loss = 0
        with torch.no_grad():
            for enc_in, dec_in, dec_out, stock_ids in valid_loader:
                enc_in, dec_in, dec_out, stock_ids = enc_in.to(Config.DEVICE), dec_in.to(Config.DEVICE), dec_out.to(Config.DEVICE), stock_ids.to(Config.DEVICE)
                outputs, _ = model(enc_in, dec_in, stock_ids)
                loss = criterion(outputs, dec_out)
                total_valid_loss += loss.item()
        avg_valid_loss = total_valid_loss / len(valid_loader)
        
        history['train_loss'].append(avg_train_loss)
        history['valid_loss'].append(avg_valid_loss)
        
        elapsed_time = time.time() - start_time
        print(f"エポック {epoch+1:02d}/{Config.EPOCHS} | 訓練損失: {avg_train_loss:.6f} | 検証損失: {avg_valid_loss:.6f} | 時間: {elapsed_time:.2f}s")
        
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            epochs_no_improve = 0
            print(f"  -> 検証損失が改善。モデルを保存: {Config.MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= Config.PATIENCE:
                print(f"\n--- 早期終了: {Config.PATIENCE}エポック連続で検証損失が改善しませんでした。---")
                break
    
    plot_loss_curves(history)

    print("\n--- テストセットで最終評価 ---")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    
    rmse, mae, mape = evaluate_and_plot(model, test_loader, scaler, total_feature_names, id_to_code)
    
    print("\n★★★ 最終テスト指標 (修正版) ★★★")
    print(f"RMSE: {rmse:.4f} (円)")
    print(f"MAE:  {mae:.4f} (円)")
    print(f"MAPE: {mape:.2f} %")
    print("★★★ 評価完了 ★★★")

if __name__ == '__main__':
    main()
