import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'

# 設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
REPORT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "case_studies_visual"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力ファイル
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
CASE_FILE = REPORT_DIR / "case_study_positive.csv" # find_best_cases.pyの出力
NEWS_FILE = DATA_DIR / "sentiment_noise_check_list.csv" # タイトル入り

def main():
    print("Generating Case Study Charts with Titles...")
    
    if not CASE_FILE.exists():
        print(f"Error: {CASE_FILE} not found. Run find_best_cases.py first.")
        return

    # データを読み込み
    df_cases = pd.read_csv(CASE_FILE, dtype={'Code': str})
    df_price = pd.read_csv(PRICE_FILE, dtype={'Code': str, 'code': str})
    if 'Code' in df_price.columns:
        df_price.rename(columns={'Code': 'code'}, inplace=True)
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    
    # 上位3件を可視化
    top_cases = df_cases.head(3)
    
    for i, row in top_cases.iterrows():
        code = row['Code'] # case file uses 'Code'
        event_date = pd.to_datetime(row['Date'])
        title = row['Title']
        score = row['News_Sentiment']
        
        print(f"Plotting Case: {code} on {event_date.date()}")
        
        # 前後2ヶ月のデータを抽出
        mask = (df_price['code'] == code) & \
               (df_price['Date'] >= event_date - pd.Timedelta(days=60)) & \
               (df_price['Date'] <= event_date + pd.Timedelta(days=60))
        df_stock = df_price[mask].sort_values('Date')
        
        if len(df_stock) == 0:
            continue
        
        # プロット
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(df_stock['Date'], df_stock['Close'], label='Stock Price', color='navy')
        
        # イベント日を強調
        price_at_event = df_stock[df_stock['Date'] == event_date]['Close']
        if not price_at_event.empty:
            price_val = price_at_event.values[0]
            ax.scatter(event_date, price_val, color='red', s=100, zorder=5)
            
            # 吹き出しでタイトルとスコアを表示
            anno_text = f"News: {title[:20]}...\nScore: {score:.2f}\nDate: {event_date.date()}"
            ax.annotate(anno_text, 
                        xy=(event_date, price_val), 
                        xytext=(event_date + pd.Timedelta(days=10), price_val * 1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05),
                        bbox=dict(boxstyle="round,pad=0.3", fc="yellow", alpha=0.3),
                        fontsize=10)

        ax.set_title(f"Event Study: {row['Name']} ({code})", fontsize=14)
        ax.set_ylabel("Price")
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"case_study_{code}_{event_date.date()}.png")
        plt.close()

    print(f"Charts saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()