import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import timedelta
import numpy as np
from pathlib import Path

# ==========================================
# 1. 設定エリア
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR1 = PROJECT_ROOT / "3_reports" / "phase3_production_deep_strict"
PRED_FILE = DATA_DIR1 / "predictions_FusionTransformer.csv"
DATA_DIR2 = PROJECT_ROOT / "1_data" / "processed"
NEWS_FILE = DATA_DIR2 / "news_sentiment_historical.csv"

# 出力画像の保存先
OUTPUT_DIR = "best_case_studies"

# 探索する候補の数
TOP_N = 5

# グラフ設定
SHIFT_DAYS = 7     # 赤線（予測）を左にずらす日数
WINDOW_DAYS = 60   # 表示期間（片側）

# 日本語フォント
plt.rcParams['font.family'] = 'MS Gothic' 
sns.set(style="whitegrid", font='MS Gothic')

def find_and_plot_best_cases():
    # ---------------------------------------------------------
    # 1. データ読み込み
    # ---------------------------------------------------------
    print("データを読み込んでいます...")
    try:
        df = pd.read_csv(PRED_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        # 予測対象日（Target_Date）を作成
        df['Target_Date'] = df['Date'] + timedelta(days=5)
    except FileNotFoundError:
        print(f"エラー: {PRED_FILE} が見つかりません。")
        return

    # ニュースデータの読み込み（オプション）
    df_news = None
    if Path(NEWS_FILE).exists():
        try:
            df_news = pd.read_csv(NEWS_FILE, low_memory=False) # DtypeWarning対策
            df_news['Date'] = pd.to_datetime(df_news['Date'])
            print("ニュースデータを読み込みました。")
        except Exception:
            print("ニュースデータの読み込みに失敗しました（スキップします）。")
    else:
        print("ニュースデータが見つかりません（スキップします）。")

    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # ---------------------------------------------------------
    # 2. 【Case 1: Gate Open】 (重要ニュースへの反応)
    # ---------------------------------------------------------
    print("\n【Case 1: Gate Open (重要ニュースへの反応)】を探索中...")
    
    candidates_open = []
    # 各銘柄の「最大Gate Scoreの日」を評価
    for code, group in df.groupby('code'):
        # 最大Gateの日を取得
        max_idx = group['Gate_Score'].idxmax()
        max_gate = group.loc[max_idx, 'Gate_Score']
        peak_date = group.loc[max_idx, 'Target_Date']
        
        # その日の前後での株価変動率（ボラティリティ）
        # 動きがないのにGateが開いているのは面白くないため
        start = peak_date - timedelta(days=10)
        end = peak_date + timedelta(days=10)
        sub = group[(group['Target_Date'] >= start) & (group['Target_Date'] <= end)]
        
        if len(sub) < 5:
            continue
        
        volatility = sub['Actual'].std() / (sub['Actual'].mean() + 1e-6)
        
        # スコア: Gateの高さ × 変動の大きさ
        score = max_gate * (1 + volatility * 10)
        
        candidates_open.append({
            'code': code,
            'name': group['Name'].iloc[0],
            'peak_date': peak_date,
            'gate_val': max_gate,
            'score': score
        })

    # 上位を取得
    df_open = pd.DataFrame(candidates_open).sort_values('score', ascending=False).head(TOP_N)
    print(f"候補銘柄: {df_open['name'].tolist()}")

    for i, (_, row) in enumerate(df_open.iterrows()):
        plot_single_case(df, df_news, row, mode="Open", rank=i+1)


    # ---------------------------------------------------------
    # 3. 【Case 2: Gate Closed】 (ノイズ遮断)
    # ---------------------------------------------------------
    print("\n【Case 2: Gate Closed (ノイズ遮断)】を探索中...")
    
    candidates_closed = []
    
    # 戦略変更: 「株価が大きく動いた日」に「Gateが閉じていた」事例を探す
    for code, group in df.groupby('code'):
        group = group.sort_values('Target_Date').reset_index(drop=True)
        
        # 株価変動（前日比の絶対値）を計算
        group['Abs_Diff'] = group['Actual'].diff().abs()
        
        # 変動が最大のトップ3日を取得してチェック
        # (最大の日だけだと、たまたまGateが開いてるかもしれないので候補を広げる)
        top_diffs = group.nlargest(3, 'Abs_Diff')
        
        for _, r in top_diffs.iterrows():
            # Gateが低い (0.28以下とする)
            if r['Gate_Score'] < 0.28:
                # スコア: 変動幅 / (Gateスコア + 小数) 
                # -> 変動が大きいのにGateが低いほど高スコア
                score = r['Abs_Diff'] / (r['Gate_Score'] + 0.01)
                
                candidates_closed.append({
                    'code': code,
                    'name': group['Name'].iloc[0],
                    'peak_date': r['Target_Date'],
                    'gate_val': r['Gate_Score'],
                    'score': score
                })
                # 1銘柄につき1候補で十分
                break
    
    if len(candidates_closed) > 0:
        df_closed = pd.DataFrame(candidates_closed).sort_values('score', ascending=False).head(TOP_N)
        print(f"候補銘柄: {df_closed['name'].tolist()}")

        for i, (_, row) in enumerate(df_closed.iterrows()):
            plot_single_case(df, df_news, row, mode="Closed", rank=i+1)
    else:
        print("条件に合うClosed事例が見つかりませんでした。閾値を調整してください。")

    print(f"\n完了: 画像は '{OUTPUT_DIR}' フォルダに保存されました。")


def plot_single_case(df, df_news, row_info, mode, rank):
    """個別の事例をプロットする関数"""
    code = row_info['code']
    name = row_info['name']
    center_date = row_info['peak_date']
    gate_val = row_info['gate_val']
    
    # データ抽出
    df_code = df[df['code'] == code].sort_values('Target_Date').reset_index(drop=True)
    
    # 期間設定 (スマートクリップ)
    start_date = center_date - timedelta(days=WINDOW_DAYS)
    end_date = center_date + timedelta(days=WINDOW_DAYS)
    
    min_date = df_code['Target_Date'].min()
    max_date = df_code['Target_Date'].max()
    if start_date < min_date:
        start_date = min_date
    if end_date > max_date:
        end_date = max_date
    
    plot_data = df_code[(df_code['Target_Date'] >= start_date) & (df_code['Target_Date'] <= end_date)].copy()
    if len(plot_data) == 0:
        return

    # 予測線のシフト用日付
    plot_data['Pred_Plot_Date'] = plot_data['Target_Date'] - timedelta(days=SHIFT_DAYS)

    # ニュース情報の取得
    news_text = "No Major News"
    if df_news is not None:
        # 中心日の前後3日以内のニュースを探す
        news_subset = df_news[
            (df_news['Code'] == str(code)) & 
            (df_news['Date'] >= center_date - timedelta(days=3)) & 
            (df_news['Date'] <= center_date + timedelta(days=3))
        ]
        if not news_subset.empty:
            # 一番スコアが極端なもの（ポジティブかネガティブ）を選ぶ
            news_subset['Abs_Sent'] = news_subset['News_Sentiment'].abs()
            top_news = news_subset.sort_values('Abs_Sent', ascending=False).iloc[0]
            
            title = top_news['Title']
            if len(title) > 15:
                title = title[:15] + "..."
            news_text = f"News: {title}\n(Sent: {top_news['News_Sentiment']:.2f})"
        else:
            if mode == "Open":
                news_text = "(Unknown High Attention Event)"
            else:
                news_text = "(No Relevant News -> Noise)"

    # --- プロット ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 上段: 株価
    ax1.plot(plot_data['Target_Date'], plot_data['Actual'], color='#333333', label='Actual Price', linewidth=1.5, alpha=0.8)
    ax1.plot(plot_data['Pred_Plot_Date'], plot_data['Pred'], color='#d62728', linestyle='--', label=f'Prediction (Shifted {SHIFT_DAYS}d)', linewidth=2.0)
    
    # 注釈
    y_center = plot_data.loc[plot_data['Target_Date'] == center_date, 'Actual'].mean()
    if not plot_data[plot_data['Target_Date'] == center_date].empty:
        y_center = plot_data.loc[plot_data['Target_Date'] == center_date, 'Actual'].values[0]
    
    ax1.axvline(center_date, color='green', linestyle=':', alpha=0.6)
    
    # 吹き出し
    ax1.annotate(
        f"{mode} Event\n{news_text}",
        xy=(center_date, y_center),
        xytext=(center_date + timedelta(days=5), y_center * 1.05),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9)
    )
    
    ax1.set_title(f"Rank {rank}: {name} ({code}) - {mode} Case", fontsize=14)
    ax1.legend(loc='upper left')
    ax1.set_xlim(plot_data['Target_Date'].min(), plot_data['Target_Date'].max())
    ax1.grid(True, alpha=0.3)

    # 下段: Gate Score
    gate_col = '#1f77b4'
    ax2.plot(plot_data['Target_Date'], plot_data['Gate_Score'], color=gate_col, label='Gate Score', linewidth=2)
    ax2.fill_between(plot_data['Target_Date'], plot_data['Gate_Score'], 0, color=gate_col, alpha=0.2)
    
    # 注目ポイントに丸印
    ax2.scatter([center_date], [gate_val], color='red', s=50, zorder=5)
    
    ax2.axvline(center_date, color='green', linestyle=':', alpha=0.6)
    
    # 閾値ライン(参考)
    ax2.axhline(0.3, color='gray', linestyle='--', linewidth=0.8)
    
    ax2.set_ylabel("Gate Score")
    ax2.set_ylim(0, 0.7) # 見やすく調整
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 日付フォーマット
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    
    save_path = f"{OUTPUT_DIR}/{mode}_Rank{rank}_{code}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"  -> 保存: {save_path}")

if __name__ == "__main__":
    find_and_plot_best_cases()
