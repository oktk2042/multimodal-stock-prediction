import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import numpy as np
import textwrap
import os

# ==========================================
# 1. 環境設定・パス定義
# ==========================================
# ディレクトリ設定
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis_30_candidates"

# 入力ファイル
NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"

# 出力ディレクトリ作成
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# プロット設定（日本語フォント）
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.grid'] = True

# ==========================================
# 2. データ読み込み・前処理関数
# ==========================================
def load_data():
    """ニュースと株価データを読み込み、整形する"""
    print("データを読み込んでいます...")
    
    # --- ニュースデータの読み込み ---
    try:
        # news_sentiment_historical.csv の列名は Date, Code, Title, News_Sentiment ... と想定
        df_news = pd.read_csv(NEWS_FILE)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Code'] = df_news['Code'].astype(str)
        print(f"ニュースデータ読み込み完了: {len(df_news)} 件")
    except Exception as e:
        print(f"エラー: ニュースファイルの読み込みに失敗しました ({NEWS_FILE})")
        raise e

    # --- 株価データの読み込み ---
    try:
        # dataset_for_modeling_top200.csv の列名は Date, code, Name, Close ... となっている
        df_price = pd.read_csv(PRICE_FILE)
        
        # 列名の統一処理 (code -> Code)
        if 'code' in df_price.columns:
            df_price = df_price.rename(columns={'code': 'Code'})
            
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price['Code'] = df_price['Code'].astype(str)
        
        print(f"株価データ読み込み完了: {len(df_price)} 件")
    except Exception as e:
        print(f"エラー: 株価ファイルの読み込みに失敗しました ({PRICE_FILE})")
        raise e
        
    return df_news, df_price

# ==========================================
# 3. 候補銘柄・期間の選定アルゴリズム
# ==========================================
def find_best_windows(df_news, df_price, window_days=90, top_n=30):
    """
    各銘柄について、ニュースが3件以上あり、かつ分析価値が高い期間を探索する。
    期間は90日（約3ヶ月）でスキャンします。
    """
    print("最適な分析対象（30銘柄）を選定中...")
    
    candidates = []
    unique_codes = df_news['Code'].unique()
    
    for code in unique_codes:
        # データ抽出
        p_data = df_price[df_price['Code'] == code].sort_values('Date')
        n_data = df_news[df_news['Code'] == code].sort_values('Date')
        
        if p_data.empty or len(n_data) < 3:
            continue
            
        # 企業名取得
        name = p_data['Name'].iloc[0] if 'Name' in p_data.columns else code

        # ニュースがある日を起点にウィンドウを評価
        # すべての日付をチェックすると遅いため、ニュース発生日の周辺を重点チェック
        check_dates = n_data['Date'].unique()
        
        best_score_for_code = -1
        best_window_for_code = None
        
        for start_date in check_dates:
            # 少し前から開始するように調整（ニュースの直前の動きも含めるため）
            window_start = start_date - pd.Timedelta(days=5)
            window_end = window_start + pd.Timedelta(days=window_days)
            
            # ウィンドウ内のデータを取得
            window_news = n_data[(n_data['Date'] >= window_start) & (n_data['Date'] <= window_end)]
            news_count = len(window_news)
            
            # 条件: ニュースが3件以上あること
            if news_count < 3:
                continue
            
            window_price = p_data[(p_data['Date'] >= window_start) & (p_data['Date'] <= window_end)]
            if window_price.empty:
                continue
                
            # --- スコアリング ---
            # 1. ニュースの密度と感情の強さ
            sentiment_magnitude = window_news['News_Sentiment'].abs().sum()
            
            # 2. 株価の変動率 (Volatility)
            price_max = window_price['Close'].max()
            price_min = window_price['Close'].min()
            volatility = (price_max - price_min) / price_min if price_min > 0 else 0
            
            # 総合スコア (変動が大きく、ニュースが多い期間を優先)
            score = (news_count * 2.0) + (sentiment_magnitude * 3.0) + (volatility * 50.0)
            
            if score > best_score_for_code:
                best_score_for_code = score
                best_window_for_code = {
                    'code': code,
                    'name': name,
                    'start': window_start,
                    'end': window_end,
                    'score': score,
                    'news_count': news_count,
                    'volatility': volatility
                }
        
        if best_window_for_code:
            candidates.append(best_window_for_code)
    
    # スコア順にソートして上位を返す
    candidates.sort(key=lambda x: x['score'], reverse=True)
    return candidates[:top_n]

# ==========================================
# 4. 可視化ロジック
# ==========================================
def plot_case_study(case, df_news, df_price, index):
    """選定されたケースのグラフを描画し保存する"""
    code = case['code']
    name = case['name']
    start_date = case['start']
    end_date = case['end']
    
    # データ抽出
    data_price = df_price[(df_price['Code'] == code) & 
                          (df_price['Date'] >= start_date) & 
                          (df_price['Date'] <= end_date)].copy()
    
    data_news = df_news[(df_news['Code'] == code) & 
                        (df_news['Date'] >= start_date) & 
                        (df_news['Date'] <= end_date)].copy()
    
    if data_price.empty:
        return

    # 図の作成
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.08)

    # --- 上段: 株価チャート ---
    ax1.plot(data_price['Date'], data_price['Close'], label='株価', color='#2C3E50', linewidth=2)
    
    # 移動平均線があれば描画（データ量による）
    if len(data_price) > 20:
        ma5 = data_price['Close'].rolling(5).mean()
        ma25 = data_price['Close'].rolling(25).mean()
        ax1.plot(data_price['Date'], ma5, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='MA(5)')
        ax1.plot(data_price['Date'], ma25, color='skyblue', linestyle='--', linewidth=1, alpha=0.7, label='MA(25)')

    # --- ニュースのアノテーション ---
    data_news = data_news.sort_values('Date')
    
    # テキスト配置の計算
    y_min, y_max = ax1.get_ylim()
    y_range = y_max - y_min
    
    # 重なり防止のため、表示高さを3段階に分散
    text_levels = [y_min + y_range * 0.93, y_min + y_range * 0.80, y_min + y_range * 0.67]
    
    # ニュースが多い場合は、スコア絶対値が高い順に最大6件に絞る（視認性確保）
    if len(data_news) > 6:
        top_indices = data_news['News_Sentiment'].abs().sort_values(ascending=False).head(6).index
        plot_news = data_news.loc[top_indices].sort_values('Date')
    else:
        plot_news = data_news

    for i, row in enumerate(plot_news.itertuples()):
        date = row.Date
        score = row.News_Sentiment
        title = str(row.Title)
        
        # 当日の株価を取得
        closest_row = data_price.iloc[(data_price['Date'] - date).abs().argsort()[:1]]
        if closest_row.empty:
            continue
        price_val = closest_row['Close'].values[0]
        
        # 色設定
        color = '#D9534F' if score >= 0 else '#5BC0DE'
        
        # 配置高さ
        level_y = text_levels[i % 3]
        
        # タイトル折り返し
        wrapped_title = "\n".join(textwrap.wrap(title, width=16))
        label_text = f"{date.strftime('%m/%d')}\n{wrapped_title}\n({score:.2f})"
        
        # アノテーション
        ax1.annotate(label_text,
                     xy=(date, price_val),
                     xytext=(date, level_y),
                     arrowprops=dict(arrowstyle="->", color='gray', connectionstyle="arc3,rad=0.1"),
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.9, linewidth=1.5),
                     fontsize=9, ha='center', va='top', zorder=10)
        
        ax1.scatter(date, price_val, color=color, s=60, zorder=5, edgecolors='white')

    ax1.set_title(f"Case {index}: {name} ({code}) - News Impact Analysis", fontsize=14, fontweight='bold')
    ax1.set_ylabel('Stock Price (JPY)')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='upper left')

    # --- 下段: 累積感情スコア ---
    merged = pd.merge(data_price[['Date']], data_news[['Date', 'News_Sentiment']], 
                      on='Date', how='left').fillna(0)
    
    # 棒グラフ（日次）
    ax2.bar(merged['Date'], merged['News_Sentiment'], color='gray', alpha=0.5, label='Daily Sentiment')
    
    # 折れ線（累積）
    merged['Cumulative'] = merged['News_Sentiment'].cumsum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged['Date'], merged['Cumulative'], color='#28a745', linewidth=2, label='Cumulative Trend')
    ax2_twin.fill_between(merged['Date'], merged['Cumulative'], 0, color='#28a745', alpha=0.1)
    
    ax2.set_ylabel('Daily Score')
    ax2_twin.set_ylabel('Cumulative Score')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 凡例
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, loc='upper left')

    # X軸
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)

    # 保存
    filename = f"case_{index:02d}_{code}.png"
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved graph: {filename}")

# ==========================================
# 5. メイン処理
# ==========================================
def main():
    # 1. データ読み込み
    try:
        df_news, df_price = load_data()
    except Exception as e:
        print(f"処理を中断します: {e}")
        return

    # 2. ケース選定
    candidates = find_best_windows(df_news, df_price, window_days=90, top_n=30)
    
    if not candidates:
        print("条件に合致するデータが見つかりませんでした。")
        return

    print("\n--- 選定された30ケースのグラフ作成を開始します ---")
    
    # リスト保存
    df_candidates = pd.DataFrame(candidates)
    df_candidates.to_csv(OUTPUT_DIR / "selected_30_candidates_list.csv", index=False)
    
    # 3. グラフ作成
    for i, case in enumerate(candidates):
        print(f"[{i+1}/{len(candidates)}] {case['name']} ({case['code']}) Score:{case['score']:.1f}")
        try:
            plot_case_study(case, df_news, df_price, i+1)
        except Exception as e:
            print(f"  Error: {e}")

    print("\n完了しました。出力フォルダを確認してください: " + str(OUTPUT_DIR))

if __name__ == "__main__":
    main()