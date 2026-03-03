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
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "analysis_30_cases_final_layout"

NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
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
    try:
        df_news = pd.read_csv(NEWS_FILE)
        df_news['Date'] = pd.to_datetime(df_news['Date'])
        df_news['Code'] = df_news['Code'].astype(str)
    except Exception as e:
        print(f"エラー: ニュースファイルの読み込みに失敗しました ({NEWS_FILE})")
        raise e

    try:
        df_price = pd.read_csv(PRICE_FILE)
        if 'code' in df_price.columns:
            df_price = df_price.rename(columns={'code': 'Code'})
        df_price['Date'] = pd.to_datetime(df_price['Date'])
        df_price['Code'] = df_price['Code'].astype(str)
    except Exception as e:
        print(f"エラー: 株価ファイルの読み込みに失敗しました ({PRICE_FILE})")
        raise e
        
    try:
        df_candidates = pd.read_csv(CANDIDATES_FILE)
        df_candidates['code'] = df_candidates['code'].astype(str)
        df_candidates['start'] = pd.to_datetime(df_candidates['start'])
        df_candidates['end'] = pd.to_datetime(df_candidates['end'])
    except Exception as e:
        print("エラー: 候補リストが見つかりません。先にgenerate_30_cases.pyを実行してください。")
        raise e

    return df_news, df_price, df_candidates

# ==========================================
# 3. 可視化ロジック (最終レイアウト調整版)
# ==========================================
def plot_case_study_final(case, df_news, df_price, index):
    code = str(case['code'])
    name = str(case['name'])
    start_date = case['start']
    end_date = case['end']
    
    # --- データ抽出 ---
    # 前後に少し余裕を持たせる
    plot_start = start_date - pd.Timedelta(days=7)
    plot_end = end_date + pd.Timedelta(days=7)
    
    data_price = df_price[(df_price['Code'] == code) & 
                          (df_price['Date'] >= plot_start) & 
                          (df_price['Date'] <= plot_end)].copy()
    
    data_news = df_news[(df_news['Code'] == code) & 
                        (df_news['Date'] >= start_date) & 
                        (df_news['Date'] <= end_date)].copy()
    
    if data_price.empty:
        return

    # --- ニュースの厳選 ---
    # 3〜5件程度に絞る。センチメントの絶対値が高い順。
    if len(data_news) > 5:
        data_news['abs_score'] = data_news['News_Sentiment'].abs()
        top_news = data_news.sort_values('abs_score', ascending=False).head(5)
        top_news = top_news.sort_values('Date')
    else:
        top_news = data_news.sort_values('Date')

    # --- プロット作成 ---
    # 高さを確保して見やすくする
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True, 
                                   gridspec_kw={'height_ratios': [3.5, 1]})
    plt.subplots_adjust(hspace=0.15, top=0.93)

    # 1. 株価チャート
    ax1.plot(data_price['Date'], data_price['Close'], label='株価', color='#003366', linewidth=2.5, zorder=2)
    
    # Y軸の範囲を意図的に広げる（吹き出し用のスペース確保）
    y_min_actual = data_price['Close'].min()
    y_max_actual = data_price['Close'].max()
    y_range = y_max_actual - y_min_actual
    
    # 上下に30%ずつの余白を追加
    y_limit_min = y_min_actual - (y_range * 0.3)
    y_limit_max = y_max_actual + (y_range * 0.35) # タイトル側は少し多めに
    ax1.set_ylim(y_limit_min, y_limit_max)

    # --- アノテーション配置ロジック ---
    # 「空白活用」アルゴリズム:
    # 株価がその期間の平均より高い -> テキストは下の余白へ
    # 株価がその期間の平均より低い -> テキストは上の余白へ
    
    price_mid = (y_max_actual + y_min_actual) / 2
    
    # 配置位置の微調整用（隣同士が被らないようにずらす係数）
    stagger_step_top = 0
    stagger_step_bottom = 0
    
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
        
        # --- 配置決定ロジック ---
        # 基本ルール: 株価が高い位置にあるなら下に、低い位置なら上に置くことで線を避ける
        if price_val > price_mid:
            # 下の余白エリアに配置
            # Y座標: 実際の最安値よりも下、かつindexで少しずらす
            base_y = y_min_actual - (y_range * 0.1)
            offset_y = - (stagger_step_bottom % 2) * (y_range * 0.12) # 交互にさらに下げる
            text_y = base_y + offset_y
            
            va = 'top' # テキストの上端を座標に合わせる
            connection_style = "arc3,rad=-0.15" # 下向きのカーブ
            stagger_step_bottom += 1
        else:
            # 上の余白エリアに配置
            # Y座標: 実際の最高値よりも上
            base_y = y_max_actual + (y_range * 0.1)
            offset_y = (stagger_step_top % 2) * (y_range * 0.12) # 交互にさらに上げる
            text_y = base_y + offset_y
            
            va = 'bottom' # テキストの下端を座標に合わせる
            connection_style = "arc3,rad=0.15" # 上向きのカーブ
            stagger_step_top += 1

        # タイトル整形: 13文字程度で折り返し
        wrapped_lines = textwrap.wrap(title, width=13)
        if len(wrapped_lines) > 3:
            wrapped_lines = wrapped_lines[:3]
            wrapped_lines[-1] += "..."
        display_text = "\n".join(wrapped_lines)
        
        label_text = f"[{date.strftime('%m/%d')}]\n{display_text}\n(Score: {score:.2f})"
        
        # アノテーション描画
        ax1.annotate(label_text,
                     xy=(date, price_val),          # 矢印の先端（株価）
                     xytext=(date, text_y),         # テキストの位置（余白エリア）
                     # xycoords='data', textcoords='data', # データ座標系を使用
                     arrowprops=dict(arrowstyle="->", color='gray', connectionstyle=connection_style, linewidth=1.2),
                     bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=color, alpha=1.0, linewidth=1.5), # alpha=1.0で背景を透過させない
                     fontsize=9, ha='center', va=va, zorder=20) # zorderを高くして最前面に
        
        # チャート上のポイント
        ax1.scatter(date, price_val, color=color, s=60, zorder=15, edgecolors='white', linewidth=1.5)

    ax1.set_title(f"Case {index}: {name} ({code}) - 重要ニュースインパクト分析", fontsize=16, fontweight='bold')
    ax1.set_ylabel('株価 (円)', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5, zorder=1) # グリッドは後ろ
    ax1.legend(loc='upper left', fontsize=10)

    # 3. 下段: 感情スコア
    merged = pd.merge(data_price[['Date']], df_news[(df_news['Code'] == code) & (df_news['Date'] >= plot_start) & (df_news['Date'] <= plot_end)][['Date', 'News_Sentiment']], 
                      on='Date', how='left').fillna(0)
    
    # 棒グラフ
    ax2.bar(merged['Date'], merged['News_Sentiment'], color='gray', alpha=0.5, width=1.5, label='日次センチメント')
    
    # 累積トレンド線
    merged['Cumulative'] = merged['News_Sentiment'].cumsum()
    ax2_twin = ax2.twinx()
    ax2_twin.plot(merged['Date'], merged['Cumulative'], color='#28a745', linewidth=2.0, label='累積感情スコア')
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
    filename = f"final_case_{index:02d}_{code}.png"
    save_path = OUTPUT_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")

# ==========================================
# 4. メイン処理
# ==========================================
def main():
    try:
        df_news, df_price, df_candidates = load_data()
    except Exception as e:
        print(f"処理を中断します: {e}")
        return

    print("\n--- 最終レイアウト版 30銘柄グラフ作成 ---")
    print(f"出力先: {OUTPUT_DIR}")
    
    for i, row in df_candidates.iterrows():
        print(f"[{i+1}/{len(df_candidates)}] Processing {row['name']} ({row['code']})...")
        try:
            plot_case_study_final(row, df_news, df_price, i+1)
        except Exception as e:
            print(f"  Error plotting {row['code']}: {e}")
            import traceback
            traceback.print_exc()

    print("\n全処理が完了しました。")

if __name__ == "__main__":
    main()