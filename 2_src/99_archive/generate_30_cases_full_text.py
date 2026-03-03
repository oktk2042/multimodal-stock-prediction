import matplotlib
# バックエンド設定（画面表示なし・ファイル出力専用）
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
# 出力先: 最終版グラフ保存フォルダ
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "thesis_charts_complete"

NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
STATS_FILE = DATA_DIR / "news_stats_by_keyword.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# グラフ設定
plt.rcParams['font.family'] = 'MS Gothic' # Windows標準
plt.rcParams['svg.fonttype'] = 'none'     # テキストを編集可能にする
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 12

# ==========================================
# 2. データ読み込み
# ==========================================
def load_data():
    print("データを読み込んでいます...")
    # ニュース詳細
    try:
        df_news = pd.read_csv(NEWS_FILE)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Code'] = df_news['Code'].astype(str)
    except Exception as e:
        print(f"エラー: {NEWS_FILE} 読み込み失敗: {e}")
        return None, None, None

    # 株価
    try:
        df_price = pd.read_csv(PRICE_FILE)
        if 'code' in df_price.columns:
            df_price = df_price.rename(columns={'code': 'Code'})
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price['Code'] = df_price['Code'].astype(str)
    except Exception as e:
        print(f"エラー: {PRICE_FILE} 読み込み失敗: {e}")
        return None, None, None

    # 銘柄統計（名称補完用）
    try:
        df_stats = pd.read_csv(STATS_FILE)
        df_stats['Code'] = df_stats['Code'].astype(str)
    except Exception:
        df_stats = pd.DataFrame()

    return df_news, df_price, df_stats

# ==========================================
# 3. 銘柄・期間選定ロジック
# ==========================================
def find_best_window_for_code(code, df_news, df_price, window_days=120, name=None):
    """ 指定銘柄の中で最も「ニュースがあり、かつ株価が動いた」期間を探す """
    p_data = df_price[df_price['Code'] == code].sort_values('Date')
    n_data = df_news[df_news['Code'] == code].sort_values('Date')
    
    if p_data.empty or len(n_data) < 3:
        return None
    
    # データ端の除外（前後60日確保のため）
    margin = pd.Timedelta(days=60)
    safe_start = p_data['Date'].min() + margin
    safe_end = p_data['Date'].max() - margin
    
    if safe_end <= safe_start:
        return None
    
    # 銘柄名
    if name is None:
        if 'Name' in p_data.columns:
            name = p_data['Name'].iloc[0]
        else:
            name = code

    # 探索対象となるニュース日付
    valid_dates = n_data[(n_data['Date'] >= safe_start) & (n_data['Date'] <= safe_end)]['Date'].unique()
    
    # 間引き（高速化）
    check_dates = valid_dates
    if len(check_dates) > 50:
        check_dates = check_dates[::len(check_dates)//50]

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
        
        # スコアリング: 変動率重視
        p_max, p_min = w_price['Close'].max(), w_price['Close'].min()
        volatility = (p_max - p_min) / p_min if p_min > 0 else 0
        sentiment = w_news['News_Sentiment'].abs().sum()
        
        # Volatility係数を大きくして「動きのある」期間を選ぶ
        score = (len(w_news) * 1.0) + (sentiment * 2.0) + (volatility * 1000.0)
        
        if score > best_score:
            best_score = score
            best_win = {
                'code': code, 'name': name,
                'start': core_start, 'end': core_end,
                'score': score, 'volatility': volatility,
                'type': 'Auto'
            }
            
    return best_win

def select_target_cases(df_news, df_price, df_stats, top_n=30):
    """ 変動重視30銘柄 + 代表銘柄を選定して統合する """
    print("分析対象を選定中...")
    
    # 1. 自動選定（全銘柄からスコア順）
    candidates = []
    all_codes = df_news['Code'].unique()
    
    for code in all_codes:
        win = find_best_window_for_code(code, df_news, df_price)
        if win:
            candidates.append(win)
        
    # スコア順にソートして上位30を取得
    candidates.sort(key=lambda x: x['score'], reverse=True)
    selected_auto = candidates[:top_n]
    
    # 2. 代表銘柄（時価総額・出来高・知名度）
    rep_codes = set()
    
    # 時価総額・出来高トップ
    if 'MarketCap' in df_price.columns:
        stats = df_price.groupby('Code')['MarketCap'].max().sort_values(ascending=False)
        rep_codes.update(stats.head(5).index.tolist())
    
    if 'Volume' in df_price.columns:
        stats_v = df_price.groupby('Code')['Volume'].mean().sort_values(ascending=False)
        rep_codes.update(stats_v.head(5).index.tolist())
        
    # 固定リスト
    famous = ['7203', '6758', '7974', '9983', '9984', '8306', '6861', '6098', '9432', '8035']
    rep_codes.update([c for c in famous if c in all_codes])
    
    # 代表銘柄のベスト期間を探す
    selected_rep = []
    for code in rep_codes:
        # 既に自動選定に含まれている場合はスキップ
        if any(c['code'] == code for c in selected_auto):
            continue
            
        win = find_best_window_for_code(code, df_news, df_price)
        if win:
            win['type'] = 'Representative'
            selected_rep.append(win)
            
    # 統合
    final_list = selected_auto + selected_rep
    print(f"選定完了: 自動選定 {len(selected_auto)}件 + 代表追加 {len(selected_rep)}件 = 計 {len(final_list)}件")
    
    return final_list

# ==========================================
# 4. 可視化ロジック (SVG・全文表示・スマート配置)
# ==========================================
def select_news_smartly(data_news, data_price):
    """ 
    重要度が高く、かつ間隔が適切なニュースを選定する 
    """
    data_price['Pct_Change'] = data_price['Close'].pct_change().abs().fillna(0)
    
    candidates = []
    for idx, row in data_news.iterrows():
        closest = data_price.iloc[(data_price['Date'] - row['Date']).abs().argsort()[:1]]
        vol = closest['Pct_Change'].values[0] if not closest.empty else 0
        
        # 重要度 = 感情スコア絶対値 * (株価変動率 + ベース)
        importance = abs(row['News_Sentiment']) * (vol + 0.01) * 100
        candidates.append({
            'index': idx, 'Date': row['Date'], 
            'importance': importance, 'score': row['News_Sentiment'],
            'title': row['Title']
        })
    
    df_cand = pd.DataFrame(candidates).sort_values('importance', ascending=False)
    if df_cand.empty:
        return data_news.head(0)

    # 上位20%を「超重要」と定義
    thresh = df_cand['importance'].quantile(0.8)
    
    selected_indices = []
    selected_dates = []
    
    for _, row in df_cand.iterrows():
        current = row['Date']
        # 超重要なら10日、それ以外は30日の間隔を要求
        min_gap = 10 if row['importance'] >= thresh else 30
        
        is_ok = True
        for d in selected_dates:
            if abs((current - d).days) < min_gap:
                is_ok = False
                break
        
        if is_ok:
            selected_indices.append(row['index'])
            selected_dates.append(current)
            
        if len(selected_indices) >= 5:
            break # 最大5件
            
    return data_news.loc[selected_indices].sort_values('Date')

def plot_thesis_chart(case, df_news, df_price, index):
    code = str(case['code'])
    name = str(case['name'])
    
    # 期間設定
    core_start = pd.to_datetime(case['start'])
    core_end = pd.to_datetime(case['end'])
    # 前後60日拡張
    view_start = core_start - pd.Timedelta(days=60)
    view_end = core_end + pd.Timedelta(days=60)
    
    # データ抽出
    data_price = df_price[(df_price['Code'] == code) & (df_price['Date'] >= view_start) & (df_price['Date'] <= view_end)].copy()
    data_news = df_news[(df_news['Code'] == code) & (df_news['Date'] >= core_start) & (df_news['Date'] <= core_end)].copy()
    
    if data_price.empty:
        return

    # ニュース選定
    plot_news = select_news_smartly(data_news, data_price)

    # --- プロット作成 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 14), sharex=True, 
                                   gridspec_kw={'height_ratios': [3.5, 1]})
    plt.subplots_adjust(hspace=0.18, top=0.92, bottom=0.08)

    # 1. 株価チャート
    ax1.plot(data_price['Date'], data_price['Close'], label='Stock Price', color='#003366', linewidth=3.0, zorder=1)
    
    # 範囲設定
    ax1.set_xlim(view_start, view_end)
    y_min, y_max = data_price['Close'].min(), data_price['Close'].max()
    y_range = y_max - y_min
    ax1.set_ylim(y_min - y_range * 0.45, y_max + y_range * 0.5) # 余白大
    
    price_mid = (y_max + y_min) / 2
    
    # 吹き出し配置用オフセット（交互配置）
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
        
        # 上下配置ロジック
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

        # 全文表示＆サイズ自動調整
        t_len = len(title)
        if t_len > 60:
            wrap_w, f_size = 24, 9
        elif t_len > 40:
            wrap_w, f_size = 20, 10
        elif t_len > 20:
            wrap_w, f_size = 18, 11
        else:
            wrap_w, f_size = 16, 12

        wrapped_title = "\n".join(textwrap.wrap(title, width=wrap_w))
        label_text = f"[{date.strftime('%m/%d')}]\n{wrapped_title}\n({score:.2f})"
        
        ax1.annotate(label_text,
                     xy=(date, price_val),
                     xytext=(date, text_y),
                     arrowprops=dict(arrowstyle="->", color='gray', connectionstyle=conn, linewidth=2.0),
                     bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=color, alpha=1.0, linewidth=2.5),
                     fontsize=f_size, fontweight='bold', ha='center', va=va, zorder=20)
        
        ax1.scatter(date, price_val, color=color, s=120, zorder=15, edgecolors='white', linewidth=2.0)

    # 分析用データは削除し、論文用のフォーマルなタイトルに変更
    title_text = f"{name} ({code}): 重要ニュースと株価トレンドの相関"
    ax1.set_title(title_text, fontsize=20, fontweight='bold', pad=15)
    
    ax1.set_ylabel('株価 (円)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=12)

    # 2. 下段: 感情スコア & トレンド
    merged = pd.merge(data_price[['Date']], 
                      df_news[(df_news['Code'] == code) & (df_news['Date'] >= view_start) & (df_news['Date'] <= view_end)][['Date', 'News_Sentiment']], 
                      on='Date', how='left').fillna(0)
    
    ax2.bar(merged['Date'], merged['News_Sentiment'], color='gray', alpha=0.5, width=2.5, label='日次感情スコア')
    
    merged['Cumulative'] = merged['News_Sentiment'].cumsum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged['Date'], merged['Cumulative'], color='#28a745', linewidth=3.0, label='累積感情スコア')
    ax2_twin.fill_between(merged['Date'], merged['Cumulative'], 0, color='#28a745', alpha=0.1)
    
    ax2.set_ylabel('感情スコア', fontsize=12)
    ax2.axhline(0, color='black', linewidth=0.8)
    ax2.grid(True, axis='y', alpha=0.3)
    ax2.set_xlim(view_start, view_end)
    
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.xticks(rotation=0, fontsize=12)

    # 保存
    filename = f"{case.get('type', 'Case')}_case_{index:02d}_{code}.svg"
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    print(f"  Saved SVG: {filename}")

# ==========================================
# 5. メイン処理
# ==========================================
def main():
    df_news, df_price, df_stats = load_data()
    if df_news is None:
        return

    candidates = select_target_cases(df_news, df_price, df_stats, top_n=30)
    if not candidates:
        print("データがありません。")
        return

    print(f"\n--- 論文用グラフ作成開始 (計 {len(candidates)} 枚) ---")
    print(f"保存先: {OUTPUT_DIR}")
    
    pd.DataFrame(candidates).to_csv(OUTPUT_DIR / "final_case_list.csv", index=False)

    for i, case in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] {case['name']} ({case['code']})")
        try:
            plot_thesis_chart(case, df_news, df_price, i+1)
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print("\n全処理が完了しました。")

if __name__ == "__main__":
    main()