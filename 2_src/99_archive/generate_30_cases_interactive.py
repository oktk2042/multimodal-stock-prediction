import matplotlib
matplotlib.use('Agg')

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np
import textwrap
import os

# ==========================================
# 1. 環境設定
# ==========================================
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis_30_candidates_large_svg"

NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['svg.fonttype'] = 'none' 
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12 

# ==========================================
# 2. データ読み込み
# ==========================================
def load_data():
    print("データを読み込んでいます...")
    try:
        df_news = pd.read_csv(NEWS_FILE)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Code'] = df_news['Code'].astype(str)
    except Exception as e:
        print(f"エラー: ニュースファイル読み込み失敗: {e}")
        return None, None

    try:
        df_price = pd.read_csv(PRICE_FILE)
        if 'code' in df_price.columns:
            df_price = df_price.rename(columns={'code': 'Code'})
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price['Code'] = df_price['Code'].astype(str)
    except Exception as e:
        print(f"エラー: 株価ファイル読み込み失敗: {e}")
        return None, None
        
    return df_news, df_price

# ==========================================
# 3. 候補選定
# ==========================================
def find_best_windows(df_news, df_price, window_days=120, top_n=30):
    print("最適な分析対象（30銘柄）を選定中...")
    candidates = []
    unique_codes = df_news['Code'].unique()
    margin_days = pd.Timedelta(days=60)
    
    for code in unique_codes:
        p_data = df_price[df_price['Code'] == code].sort_values('Date')
        n_data = df_news[df_news['Code'] == code].sort_values('Date')
        
        if p_data.empty or len(n_data) < 3:
            continue
        
        data_min = p_data['Date'].min()
        data_max = p_data['Date'].max()
        
        safe_start = data_min + margin_days
        safe_end = data_max - margin_days
        
        if safe_end <= safe_start:
            continue
            
        name = p_data['Name'].iloc[0] if 'Name' in p_data.columns else code
        
        valid_dates = n_data[(n_data['Date'] >= safe_start) & (n_data['Date'] <= safe_end)]['Date'].unique()
        check_dates = valid_dates[::len(valid_dates)//50] if len(valid_dates) > 100 else valid_dates

        best_score = -1
        best_win = None
        
        for start_date in check_dates:
            core_start = start_date - pd.Timedelta(days=5)
            core_end = core_start + pd.Timedelta(days=window_days)
            
            if core_end > safe_end:
                continue
            
            w_news = n_data[(n_data['Date'] >= core_start) & (n_data['Date'] <= core_end)]
            if len(w_news) < 3:
                continue
            
            w_price = p_data[(p_data['Date'] >= core_start) & (p_data['Date'] <= core_end)]
            if w_price.empty:
                continue
            
            p_max, p_min = w_price['Close'].max(), w_price['Close'].min()
            volatility = (p_max - p_min) / p_min if p_min > 0 else 0
            sentiment_impact = w_news['News_Sentiment'].abs().sum()
            
            score = (len(w_news) * 1.0) + (sentiment_impact * 2.0) + (volatility * 1000.0)
            
            if score > best_score:
                best_score = score
                best_win = {
                    'code': code, 'name': name, 
                    'start': core_start, 'end': core_end, 
                    'score': score, 'volatility': volatility
                }
        
        if best_win:
            candidates.append(best_win)
    
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]

# ==========================================
# 4. 可視化ロジック (大型フォント・SVG)
# ==========================================
def select_news_smartly(data_news, data_price):
    data_price['Pct_Change'] = data_price['Close'].pct_change().abs().fillna(0)
    
    candidates = []
    for idx, row in data_news.iterrows():
        closest = data_price.iloc[(data_price['Date'] - row['Date']).abs().argsort()[:1]]
        vol = closest['Pct_Change'].values[0] if not closest.empty else 0
        importance = abs(row['News_Sentiment']) * (vol + 0.01) * 100
        candidates.append({
            'index': idx, 'Date': row['Date'], 
            'importance': importance, 'score': row['News_Sentiment'],
            'title': row['Title']
        })
    
    df_cand = pd.DataFrame(candidates).sort_values('importance', ascending=False)
    if df_cand.empty:
        return data_news.head(0)

    high_imp_thresh = df_cand['importance'].quantile(0.8)
    selected_indices = []
    selected_dates = []
    
    for _, row in df_cand.iterrows():
        current = row['Date']
        is_high = row['importance'] >= high_imp_thresh
        min_gap = 10 if is_high else 30 
        
        is_ok = True
        for d in selected_dates:
            if abs((current - d).days) < min_gap:
                is_ok = False
                break
        
        if is_ok:
            selected_indices.append(row['index'])
            selected_dates.append(current)
            
        if len(selected_indices) >= 5:
            break
            
    return data_news.loc[selected_indices].sort_values('Date')

def plot_case_study_large_svg(case, df_news, df_price, index):
    code = str(case['code'])
    name = str(case['name'])
    
    core_start = pd.to_datetime(case['start'])
    core_end = pd.to_datetime(case['end'])
    view_start = core_start - pd.Timedelta(days=60)
    view_end = core_end + pd.Timedelta(days=60)
    
    data_price = df_price[(df_price['Code'] == code) & (df_price['Date'] >= view_start) & (df_price['Date'] <= view_end)].copy()
    data_news = df_news[(df_news['Code'] == code) & (df_news['Date'] >= core_start) & (df_news['Date'] <= core_end)].copy()
    
    if data_price.empty:
        return

    plot_news = select_news_smartly(data_news, data_price)

    # --- プロット作成 (サイズ大きめ) ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, gridspec_kw={'height_ratios': [3.5, 1]})
    plt.subplots_adjust(hspace=0.18, top=0.92, bottom=0.08)

    # 1. 株価チャート
    ax1.plot(data_price['Date'], data_price['Close'], label='Stock Price', color='#003366', linewidth=3.0, zorder=1)
    
    ax1.set_xlim(view_start, view_end)
    y_min, y_max = data_price['Close'].min(), data_price['Close'].max()
    y_range = y_max - y_min
    ax1.set_ylim(y_min - y_range * 0.45, y_max + y_range * 0.5) # 余白さらに拡大
    
    price_mid = (y_max + y_min) / 2
    
    offsets_top = [0.18, 0.35]
    offsets_bottom = [0.18, 0.35]
    idx_top = 0
    idx_bottom = 0
    
    for i, row in enumerate(plot_news.itertuples()):
        date = row.Date
        score = row.News_Sentiment
        title = str(row.Title)
        
        closest = data_price.iloc[(data_price['Date'] - date).abs().argsort()[:1]]
        if closest.empty:
            continue
        price_val = closest['Close'].values[0]
        
        color = '#D9534F' if score >= 0 else '#5BC0DE'
        
        if price_val > price_mid:
            offset = offsets_bottom[idx_bottom % 2]
            text_y = y_min - (y_range * offset)
            va, conn = 'top', f"arc3,rad={-0.2}" 
            idx_bottom += 1
        else:
            offset = offsets_top[idx_top % 2]
            text_y = y_max + (y_range * offset)
            va, conn = 'bottom', f"arc3,rad={0.2}"
            idx_top += 1

        wrapped_title = "\n".join(textwrap.wrap(title, width=12)[:3])
        # 文字サイズ 11pt, 太字
        label_text = f"[{date.strftime('%m/%d')}]\n{wrapped_title}\n({score:.2f})"
        
        ax1.annotate(label_text,
                     xy=(date, price_val),
                     xytext=(date, text_y),
                     arrowprops=dict(arrowstyle="->", color='gray', connectionstyle=conn, linewidth=2.0),
                     bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=color, alpha=1.0, linewidth=2.5), # 枠線太く、余白広く
                     fontsize=11, fontweight='bold', ha='center', va=va, zorder=20)
        
        ax1.scatter(date, price_val, color=color, s=120, zorder=15, edgecolors='white', linewidth=2.0)

    ax1.set_title(f"Case {index}: {name} ({code}) - Volatility: {case['volatility']:.2f}", fontsize=18, fontweight='bold')
    ax1.set_ylabel('Price (JPY)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=12)

    # 2. 下段
    merged = pd.merge(data_price[['Date']], 
                      df_news[(df_news['Code'] == code) & (df_news['Date'] >= view_start) & (df_news['Date'] <= view_end)][['Date', 'News_Sentiment']], 
                      on='Date', how='left').fillna(0)
    
    ax2.bar(merged['Date'], merged['News_Sentiment'], color='gray', alpha=0.5, width=2.5, label='Daily Sentiment')
    
    merged['Cumulative'] = merged['News_Sentiment'].cumsum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged['Date'], merged['Cumulative'], color='#28a745', linewidth=3.0, label='Cumulative Trend')
    ax2_twin.fill_between(merged['Date'], merged['Cumulative'], 0, color='#28a745', alpha=0.1)
    
    ax2.set_ylabel('Sentiment Score', fontsize=12)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_xlim(view_start, view_end)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=0, fontsize=12)

    filename = f"large_case_{index:02d}_{code}.svg"
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"  Saved SVG: {filename}")

# ==========================================
# 5. メイン処理
# ==========================================
def main():
    df_news, df_price = load_data()
    if df_news is None:
        return

    candidates = find_best_windows(df_news, df_price, window_days=120, top_n=30)
    if not candidates:
        print("データがありません。")
        return

    print("\n--- 30銘柄グラフ作成 (文字拡大・SVG版) ---")
    print(f"保存先: {OUTPUT_DIR}")
    
    pd.DataFrame(candidates).to_csv(OUTPUT_DIR / "selected_candidates_large.csv", index=False)

    for i, case in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] {case['name']} ({case['code']}) Vol:{case['volatility']:.2f}")
        try:
            plot_case_study_large_svg(case, df_news, df_price, i+1)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n全処理が完了しました。")

if __name__ == "__main__":
    main()