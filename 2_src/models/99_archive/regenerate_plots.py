import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import os

# ==========================================
# 1. 日本語文字化け対策 (Windows用)
# ==========================================
import matplotlib
matplotlib.rcParams['font.family'] = 'MS Gothic'
matplotlib.rcParams['axes.unicode_minus'] = False
sns.set(font='MS Gothic')

# ==========================================
# 2. 設定
# ==========================================
# 出力ディレクトリ
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR   = PROJECT_ROOT / "3_reports" / "phase3_production"

print(f"参照ディレクトリ: {OUTPUT_DIR}")

if not OUTPUT_DIR.exists():
    print(f"エラー: ディレクトリが見つかりません: {OUTPUT_DIR}")
    exit()

# ==========================================
# 3. 各種プロット関数
# ==========================================

def plot_model_comparison(output_dir):
    """モデル比較グラフの再生成"""
    csv_path = output_dir / "model_comparison_summary.csv"
    if not csv_path.exists():
        print("スキップ: model_comparison_summary.csv が見つかりません")
        return

    print("処理中: モデル比較グラフ...")
    df = pd.read_csv(csv_path)

    # Accuracy比較
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='Accuracy', data=df)
    plt.title("モデル精度比較 (Accuracy)") # 日本語化
    plt.ylim(40, 60)
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y', alpha=0.3)
    plt.savefig(output_dir / "model_accuracy_comparison.png")
    plt.close()

    # R2_Return比較
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Model', y='R2_Return', data=df)
    plt.title("モデル予測精度比較 (R2 Score - Return)") # 日本語化
    plt.ylabel("R2 Score")
    plt.grid(axis='y', alpha=0.3)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.savefig(output_dir / "model_return_r2_comparison.png")
    plt.close()

def plot_predictions(output_dir):
    """時系列予測プロットの再生成"""
    # predictions_*.csv をすべて検索
    files = list(output_dir.glob("predictions_*.csv"))
    
    for csv_path in files:
        model_name = csv_path.stem.replace("predictions_", "")
        print(f"処理中: {model_name} の予測プロット...")
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # ランダムに3銘柄選んでプロット
        codes = df['code'].unique()
        if len(codes) == 0: continue
        
        # 毎回違うと困る場合は固定するか、全銘柄やるかですが、今回はランダム3件
        samples = np.random.choice(codes, min(3, len(codes)), replace=False)

        # 1. Full期間
        fig, axes = plt.subplots(len(samples), 1, figsize=(10, 4*len(samples)), sharex=False)
        if len(samples) == 1: axes = [axes]

        for ax, code in zip(axes, samples):
            data = df[df['code'] == code].sort_values('Date')
            name = data['Name'].iloc[0] if 'Name' in data.columns else code
            
            ax.plot(data['Date'], data['Actual'], label='実測値 (Actual)', color='black', alpha=0.6)
            ax.plot(data['Date'], data['Pred'], label='予測値 (Prediction)', color='red', linestyle='--', alpha=0.8)
            ax.set_title(f"{model_name}: {code} - {name} (全期間)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_pred_full.png")
        plt.close()

        # 2. Zoom (直近100日)
        fig, axes = plt.subplots(len(samples), 1, figsize=(10, 4*len(samples)), sharex=False)
        if len(samples) == 1: axes = [axes]

        for ax, code in zip(axes, samples):
            data = df[df['code'] == code].sort_values('Date').iloc[-100:]
            if len(data) == 0: continue
            name = data['Name'].iloc[0] if 'Name' in data.columns else code

            ax.plot(data['Date'], data['Actual'], label='実測値', color='black', marker='.', alpha=0.6)
            ax.plot(data['Date'], data['Pred'], label='予測値', color='red', linestyle='--', marker='.', alpha=0.8)
            ax.set_title(f"{model_name}: {code} - {name} (直近100日)")
            ax.legend()
            ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_pred_zoom.png")
        plt.close()

def plot_scatter(output_dir):
    """散布図の再生成"""
    files = list(output_dir.glob("predictions_*.csv"))
    for csv_path in files:
        model_name = csv_path.stem.replace("predictions_", "")
        print(f"処理中: {model_name} の散布図...")
        
        df = pd.read_csv(csv_path)
        
        # リターン計算
        actual_ret = (df['Actual'] - df['Current']) / df['Current']
        pred_ret = df['Pred_Return']

        plt.figure(figsize=(6, 6))
        plt.scatter(actual_ret, pred_ret, alpha=0.3, s=10)

        # 基準線
        max_val = max(actual_ret.max(), pred_ret.max())
        min_val = min(actual_ret.min(), pred_ret.min())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='理想線')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)

        plt.title(f"{model_name}: リターン予測散布図")
        plt.xlabel("実測リターン (Actual)")
        plt.ylabel("予測リターン (Predicted)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_scatter.png")
        plt.close()

def plot_feature_importance(output_dir):
    """特徴量重要度の再生成"""
    files = list(output_dir.glob("*_feature_importance.csv"))
    for csv_path in files:
        model_name = csv_path.stem.replace("_feature_importance", "")
        print(f"処理中: {model_name} の特徴量重要度...")
        
        df = pd.read_csv(csv_path)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(x='Importance', y='Feature', data=df.head(20))
        plt.title(f'{model_name} 特徴量重要度 (Top 20)')
        plt.xlabel("重要度 (Importance)")
        plt.tight_layout()
        plt.savefig(output_dir / f"{model_name}_feature_importance.png")
        plt.close()

# ==========================================
# メイン実行
# ==========================================
if __name__ == "__main__":
    print("--- 画像再生成ツール (日本語フォント対応版) ---")
    
    # 1. サマリー比較 (Accuracy, R2)
    plot_model_comparison(OUTPUT_DIR)
    
    # 2. 予測プロット (Full, Zoom)
    plot_predictions(OUTPUT_DIR)
    
    # 3. 散布図
    plot_scatter(OUTPUT_DIR)
    
    # 4. 特徴量重要度
    plot_feature_importance(OUTPUT_DIR)
    
    print("\nすべての画像の再生成が完了しました。")
    print(f"保存先: {OUTPUT_DIR}")