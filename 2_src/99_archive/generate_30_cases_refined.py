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
# 出力先を少し変えて区別します
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis_30_cases_refined"

NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
# 先ほど生成された候補リストを使用
CANDIDATES_FILE = PROJECT_ROOT / "3_reports" / "analysis_30_candidates" / "selected_30_candidates_list.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定
plt.rcParams['font.family'] = 'MS Gothic'
plt.rcParams['axes.grid'] = True

# ==========================================
# 2. データ読み込み関数
# ==========================================
def load_data():
    print("データを読み込んでいます...")
    
    # ニュースデータ
    try:
        df_news = pd.read_csv(NEWS_FILE)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Code'] = df_news['Code'].astype(str)
    except Exception as e:
        print(f"エラー: ニュースファイルの読み込みに失敗しました ({NEWS_FILE})")
        raise e

    # 株価データ
    try:
        df_price = pd.read_csv(PRICE_FILE)
        if 'code' in df_price.columns:
            df_price = df_price.rename(columns={'code': 'Code'})
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price['Code'] = df_price['Code'].astype(str)
    except Exception as e:
        print(f"エラー: 株価ファイルの読み込みに失敗しました ({PRICE_FILE})")
        raise e
        
    # 候補リスト
    try:
        df_candidates = pd.read_csv(CANDIDATES_FILE)
        df_candidates['code'] = df_candidates['code'].astype(str)
        df_candidates['start'] = pd.to_datetime(df_candidates['start'])
        df_candidates['end'] = pd.to_datetime(df_candidates['end'])
        print(f"分析対象リスト読み込み完了: {len(df_candidates)} 件")
    except Exception as e:
        print(f"エラー: 候補リストが見つかりません。先にgenerate_30_cases.pyを実行してください。({CANDIDATES_FILE})")
        raise e

    return df_news, df_price, df_candidates

# ==========================================
# 3. 可視化ロジック (改良版)
# ==========================================
def plot_case_study_refined(case, df_news, df_price, index):
    code = str(case['code'])
    name = str(case['name'])
    start_date = case['start']
    end_date = case['end']
    
    # --- データ抽出 ---
    # 前後に少し余裕を持たせる（グラフの見た目のため）
    plot_start = start_date - pd.Timedelta(days=10)
    plot_end = end_date + pd.Timedelta(days=10)
    
    data_price = df_price[(df_price['Code'] == code) & 
                          (df_price['Date'] >= plot_start) & 
                          (df_price['Date'] <= plot_end)].copy()
    
    # ニュースは厳密に期間内で抽出
    data_news = df_news[(df_news['Code'] == code) & 
                        (df_news['Date'] >= start_date) & 
                        (df_news['Date'] <= end_date)].copy()
    
    if data_price.empty:
        print(f"Skip: 株価データなし ({code})")
        return

    # --- ニュースの厳選 (Filtering) ---
    # センチメントの絶対値が大きい順に上位5件を取得
    # これにより「重要度が高い」と思われるニュースのみに絞る
    if len(data_news) > 5:
        data_news['abs_score'] = data_news['News_Sentiment'].abs()
        top_news = data_news.sort_values('abs_score', ascending=False).head(5)
        top_news = top_news.sort_values('Date') # 日付順に戻す
    else:
        top_news = data_news.sort_values('Date')

    # --- プロット作成 ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, 
                                   gridspec_kw={'height_ratios': [3, 1]})
    plt.subplots_adjust(hspace=0.08)

    # 1. 株価チャート
    ax1.plot(data_price['Date'], data_price['Close'], label='株価', color='#2C3E50', linewidth=2.5)
    
    # 2. アノテーション配置の工夫
    # Y軸の範囲を取得
    y_min, y_max = data_price['Close'].min(), data_price['Close'].max()
    y_range = y_max - y_min
    
    # テキスト配置の戦略:
    # ニュースA: 上に吹き出し
    # ニュースB: 下に吹き出し
    # これを交互に行うことで重なりを劇的に減らす
    
    last_annotate_x = None # 直前の注釈のX座標
    
    for i, row in enumerate(top_news.itertuples()):
        date = row.Date
        score = row.News_Sentiment
        title = str(row.Title)
        
        # 当日の株価
        closest = data_price.iloc[(data_price['Date'] - date).abs().argsort()[:1]]
        if closest.empty:
            continue
        price_val = closest['Close'].values[0]
        
        # 色: ポジティブ(赤), ネガティブ(青)
        color = '#D9534F' if score >= 0 else '#5BC0DE'
        
        # 上下に振り分けるロジック
        # iが偶数なら上、奇数なら下 (基本方針)
        # ただし、株価がその期間の高値圏なら下、安値圏なら上に出す方が見やすい場合もある
        # ここではシンプルに「交互配置」を採用しつつ、矢印の長さを変える
        
        is_upper = (i % 2 == 0)
        
        if is_upper:
            # 上に配置
            xytext_offset = (0, 60 + (i % 2)*20) # Y方向にずらす
            va = 'bottom'
            connection_style = "arc3,rad=0.2"
        else:
            # 下に配置
            xytext_offset = (0, -60 - (i % 2)*20)
            va = 'top'
            connection_style = "arc3,rad=-0.2"

        # タイトル整形: 14文字で折り返し、最大3行まで
        wrapped_lines = textwrap.wrap(title, width=14)
        if len(wrapped_lines) > 3:
            wrapped_lines = wrapped_lines[:3]
            wrapped_lines[-1] += "..."
        display_text = "\n".join(wrapped_lines)
        
        label_text = f"[{date.strftime('%m/%d')}]\n{display_text}\n(Score: {score:.2f})"
        
        # アノテーション描画
        ax1.annotate(label_text,
                     xy=(date, price_val),
                     xytext=xytext_offset,
                     textcoords='offset points', # ポイント単位の相対移動
                     arrowprops=dict(arrowstyle="->", color='gray', connectionstyle=connection_style, linewidth=1.5),
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=0.95, linewidth=2),
                     fontsize=10, ha='center', va=va, zorder=20)
        
        # チャート上のポイント
        ax1.scatter(date, price_val, color=color, s=80, zorder=15, edgecolors='white', linewidth=1.5)

    ax1.set_title(f"Case {index}: {name} ({code}) - 重要ニュースと株価インパクト", fontsize=16, fontweight='bold')
    ax1.set_ylabel('株価 (円)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='upper left', fontsize=10)

    # 3. 下段: 感情スコア
    # 期間全体のニュースを使うか、厳選したニュースだけを使うか？
    # -> トレンドを見るために下段は「期間内の全ニュース」を表示する
    merged = pd.merge(data_price[['Date']], df_news[(df_news['Code'] == code) & (df_news['Date'] >= plot_start) & (df_news['Date'] <= plot_end)][['Date', 'News_Sentiment']], 
                      on='Date', how='left').fillna(0)
    
    # 棒グラフ
    ax2.bar(merged['Date'], merged['News_Sentiment'], color='gray', alpha=0.6, width=1.5, label='日次センチメント')
    
    # 累積トレンド線
    merged['Cumulative'] = merged['News_Sentiment'].cumsum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged['Date'], merged['Cumulative'], color='#28a745', linewidth=2.5, label='累積感情スコア')
    ax2_twin.fill_between(merged['Date'], merged['Cumulative'], 0, color='#28a745', alpha=0.1)
    
    ax2.set_ylabel('日次スコア', fontsize=10)
    ax2_twin.set_ylabel('累積スコア', fontsize=10)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.grid(True, axis='y', alpha=0.3)
    
    # 凡例
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(h1+h2, l1+l2, loc='upper left', fontsize=9)

    # X軸フォーマット
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y/%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=0)

    # 保存
    filename = f"refined_case_{index:02d}_{code}.png"
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    # 1. データ読み込み
    try:
        df_news, df_price, df_candidates = load_data()
    except Exception as e:
        print(f"処理を中断します: {e}")
        return

    print("\n--- 選定済み30銘柄のグラフ作成 (Refined Version) ---")
    print(f"出力先: {OUTPUT_DIR}")
    
    for i, row in df_candidates.iterrows():
        print(f"[{i+1}/{len(df_candidates)}] Processing {row['name']} ({row['code']})...")
        try:
            plot_case_study_refined(row, df_news, df_price, i+1)
        except Exception as e:
            print(f"  Error plotting {row['code']}: {e}")

    print("\n全処理が完了しました。")

if __name__ == "__main__":
    main()