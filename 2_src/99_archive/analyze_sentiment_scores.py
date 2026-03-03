import pandas as pd
from pathlib import Path

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル (スコア化済みの巨大CSV)
INPUT_FILE = DATA_DIR / "news_sentiment_historical.csv"

# 出力ディレクトリ
OUTPUT_DIR = DATA_DIR / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"--- 感情スコア分析開始: {INPUT_FILE.name} ---")
    
    if not INPUT_FILE.exists():
        print("エラー: 入力ファイルが見つかりません。")
        return

    # 1. データの読み込み
    try:
        # 必要な列だけ読み込む
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return

    print(f"データ件数: {len(df):,} 行")
    
    # 型変換
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date', 'News_Sentiment'])
    df['Code'] = df['Code'].astype(str)

    # 2. 基本統計量
    print("\n[1] スコア全体の基本統計:")
    stats = df['News_Sentiment'].describe()
    print(stats)
    
    # ヒストグラム用データ (簡易)
    bins = [-1.0, -0.5, -0.1, 0.1, 0.5, 1.0]
    labels = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive']
    df['Sentiment_Label'] = pd.cut(df['News_Sentiment'], bins=bins, labels=labels)
    print("\n分布:")
    print(df['Sentiment_Label'].value_counts(normalize=True))

    # 3. ノイズチェックリスト作成 (重要)
    # 各銘柄で「スコアの絶対値が高い（影響力が大きい）記事」を抽出して、中身が変じゃないか確認用
    print("\n[2] ノイズチェック用リストを作成中...")
    
    # 絶対値が高い順にソート
    df['Abs_Score'] = df['News_Sentiment'].abs()
    df_sorted = df.sort_values('Abs_Score', ascending=False)
    
    # 各銘柄からトップ1記事ずつサンプリング (最大500銘柄)
    # これを見ることで「鹿島アントラーズ」や「ホワイトハウス」のような誤爆がないか確認できる
    noise_check_df = df_sorted.groupby('Code').head(1)
    noise_check_df = noise_check_df[['Date', 'Code', 'Keyword', 'Title', 'News_Sentiment']].sort_values('Code')
    
    noise_file = OUTPUT_DIR / "sentiment_noise_check_list.csv"
    noise_check_df.to_csv(noise_file, index=False, encoding='utf-8-sig')

    # 4. 銘柄別スコア平均 (ランキング)
    print("[3] 銘柄別 平均スコアランキング作成中...")
    stock_scores = df.groupby('Code')['News_Sentiment'].agg(['mean', 'count', 'std', 'min', 'max'])
    stock_scores = stock_scores.sort_values('mean', ascending=False)
    
    ranking_file = OUTPUT_DIR / "sentiment_ranking_by_code.csv"
    stock_scores.to_csv(ranking_file, encoding='utf-8-sig')

    # 5. 時系列推移 (マクロ vs 個別平均)
    print("[4] 月次推移データの作成中...")
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # マクロ (9999)
    macro_monthly = df[df['Code'] == '9999'].groupby('YearMonth')['News_Sentiment'].mean()
    
    # 個別銘柄 (9999以外) の全体平均
    individual_monthly = df[df['Code'] != '9999'].groupby('YearMonth')['News_Sentiment'].mean()
    
    monthly_trend = pd.DataFrame({
        'Macro_Sentiment (9999)': macro_monthly,
        'Individual_Avg_Sentiment': individual_monthly
    })
    
    trend_file = OUTPUT_DIR / "sentiment_monthly_trend.csv"
    monthly_trend.to_csv(trend_file, encoding='utf-8-sig')

    print("\n--- 分析完了 ---")
    print("以下のファイルを確認して、ノイズ（無関係な記事）がないかチェックしてください。")
    print(f"1. ノイズチェック表: {noise_file}")
    print(f"2. 銘柄別スコア: {ranking_file}")
    print(f"3. 月次トレンド: {trend_file}")
    
    print("\n★特に 'sentiment_noise_check_list.csv' を見て、")
    print("  'Keyword' と 'Title' が全く関係ない銘柄がないか確認することをお勧めします。")

if __name__ == "__main__":
    main()