import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
NEWS_FILE = DATA_DIR / "sentiment_noise_check_list.csv"

# MS Gothicを指定
plt.rcParams['font.family'] = 'MS Gothic'

# ★修正版ターゲットリスト (全てTop200に含まれる銘柄)
TARGETS = [
    # 1. Positive Main: 日産自動車 (7201) - 販売増などの好材料
    {"code": "7201", "name": "Nissan Motor", "start": "2024-01-01", "end": "2024-03-31", "desc": "Positive: Strong Sales"},
    
    # 2. Negative Main: 資生堂 (4911) - 業績懸念での下落
    {"code": "4911", "name": "Shiseido",     "start": "2024-01-01", "end": "2024-05-31", "desc": "Negative: Earnings Concern"},
    
    # 3. Appendix: SUBARU (7270) - 成功済み
    {"code": "7270", "name": "SUBARU",       "start": "2024-04-01", "end": "2024-06-30", "desc": "Positive: Strong Earnings"},
    
    # 4. Appendix: ZOZO (3092) - 成功済み
    {"code": "3092", "name": "ZOZO",         "start": "2024-10-01", "end": "2024-12-31", "desc": "Positive: Sales Growth"},
    
    # 5. Appendix: Canon (7751) - 成功済み
    {"code": "7751", "name": "Canon",        "start": "2024-01-01", "end": "2024-03-31", "desc": "Positive: Profit Increase"},
]

def plot_case_study(target, df_price_all, df_news_all):
    code = target['code']
    name = target['name']
    start_date = target['start']
    end_date = target['end']
    
    print(f"Processing {name} ({code})...")
    
    # データ抽出
    mask_price = (df_price_all['code'] == code) & (df_price_all['Date'] >= start_date) & (df_price_all['Date'] <= end_date)
    data_price = df_price_all[mask_price].sort_values('Date').reset_index(drop=True)
    
    mask_news = (df_news_all['Code'] == code) & (df_news_all['Date'] >= start_date) & (df_news_all['Date'] <= end_date)
    data_news = df_news_all[mask_news].sort_values('Date')
    
    if data_price.empty:
        print(f"  No price data for {code}. Skipping.")
        return

    # プロット作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.1)
    
    # 上段: 株価
    ax1.plot(data_price['Date'], data_price['Close'], label='Stock Price', color='#003366', linewidth=2)
    ax1.set_title(f"Case Study: {name} ({code}) - {target['desc']}", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Price (JPY)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # ニュース注釈 (Top 2 events)
    if not data_news.empty:
        # スコア絶対値が高い順
        top_news = data_news.reindex(data_news['News_Sentiment'].abs().sort_values(ascending=False).index).head(2)
        for _, row in top_news.iterrows():
            date = row['Date']
            score = row['News_Sentiment']
            title = str(row['Title'])[:20] + "..."
            
            # 株価取得
            price_row = data_price[data_price['Date'] == date]
            if not price_row.empty:
                price = price_row.iloc[0]['Close']
                color = '#D9534F' if score > 0 else '#5BC0DE'
                
                # 注釈
                ax1.annotate(f"{title}\n(Score: {score:.2f})", 
                             xy=(date, price), xytext=(date, price * 1.1),
                             arrowprops=dict(arrowstyle="->", color=color, lw=2),
                             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.9),
                             fontsize=9, ha='center')

    # 下段: 感情スコア
    # マージして全日付用意
    merged = pd.merge(data_price[['Date']], data_news[['Date', 'News_Sentiment']], on='Date', how='left').fillna(0)
    colors = ['#D9534F' if x >= 0 else '#5BC0DE' for x in merged['News_Sentiment']]
    ax2.bar(merged['Date'], merged['News_Sentiment'], color=colors, alpha=0.8)
    ax2.set_ylabel('FinBERT Score')
    ax2.set_ylim(-1.0, 1.0)
    ax2.axhline(0, color='black', lw=0.5)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # フォーマット
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
    plt.xticks(rotation=45)
    
    # 保存
    save_path = OUTPUT_DIR / f"case_study_{code}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")

def main():
    # データ一括ロード
    if not PRICE_FILE.exists() or not NEWS_FILE.exists():
        print("Data files missing.")
        return
        
    df_price = pd.read_csv(PRICE_FILE, dtype={'code': str})
    df_price['Date'] = pd.to_datetime(df_price['Date'])
    
    df_news = pd.read_csv(NEWS_FILE, dtype={'Code': str})
    df_news['Date'] = pd.to_datetime(df_news['Date'])
    
    for target in TARGETS:
        plot_case_study(target, df_price, df_news)

if __name__ == "__main__":
    main()