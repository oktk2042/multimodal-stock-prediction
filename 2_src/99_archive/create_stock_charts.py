import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib # 日本語の文字化けを防ぐ
from pathlib import Path
import warnings

# MatplotlibのUserWarningを非表示にする
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. 設定 ---
# 入力するCSVファイルのパス
INPUT_CSV_PATH = Path("C:/M2_Research_Project/1_data/processed/final_model_input_dataset.csv")

# グラフを保存するフォルダのパス
OUTPUT_CHART_DIR = Path("C:/M2_Research_Project/1_data/stock_charts")

# --- 2. メイン処理 ---
def generate_all_stock_charts():
    """
    CSVファイルを読み込み、含まれる全銘柄の株価グラフを個別のPNGファイルとして保存する。
    """
    # --- データの読み込み ---
    try:
        print(f"--- データファイルを読み込んでいます: {INPUT_CSV_PATH} ---")
        # 'Date'列を日付型として読み込む
        df = pd.read_csv(INPUT_CSV_PATH, parse_dates=['Date'])
        print("--- データの読み込み完了 ---")
    except FileNotFoundError:
        print(f"[エラー] ファイルが見つかりません: {INPUT_CSV_PATH}")
        print("ファイルパスが正しいか、または前処理が完了しているか確認してください。")
        return

    # --- 保存用フォルダの作成 ---
    OUTPUT_CHART_DIR.mkdir(parents=True, exist_ok=True)
    print(f"--- グラフの保存先フォルダ: {OUTPUT_CHART_DIR} ---")

    # --- 全銘柄コードの取得 ---
    unique_codes = sorted(df['Code'].unique())
    total_stocks = len(unique_codes)
    print(f"--- {total_stocks}銘柄のグラフを生成します ---")

    # --- 各銘柄のグラフをループで生成 ---
    for i, code in enumerate(unique_codes):
        # 銘柄ごとのデータを抽出
        stock_df = df[df['Code'] == code].sort_values('Date')
        
        # 銘柄名を取得（データ内に存在する場合）
        stock_name = stock_df['name'].iloc[0] if 'name' in stock_df.columns else ''

        # --- グラフ描画 ---
        plt.figure(figsize=(15, 8))
        plt.plot(stock_df['Date'], stock_df['Close'], label=f'{code} Close Price')
        
        # グラフの装飾
        plt.title(f"{stock_name} ({code}) の株価推移 (2020-2025)", fontsize=18)
        plt.xlabel("日付 (Date)", fontsize=12)
        plt.ylabel("終値 (円)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        # y軸の数値をカンマ区切りにするフォーマッター
        ax = plt.gca()
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
        
        plt.tight_layout() # レイアウトを自動調整

        # --- ファイルとして保存 ---
        output_path = OUTPUT_CHART_DIR / f"chart_{code}.png"
        plt.savefig(output_path)
        
        # メモリを解放するためにプロットを閉じる
        plt.close()
        
        # 進捗状況を表示
        if (i + 1) % 10 == 0 or (i + 1) == total_stocks:
            print(f"  {i + 1}/{total_stocks} 銘柄完了: chart_{code}.png を保存しました。")

    print(f"\n--- 全{total_stocks}銘柄のグラフ生成が完了しました。 ---")

# --- 3. スクリプトの実行 ---
if __name__ == '__main__':
    generate_all_stock_charts()
