import pandas as pd
from pathlib import Path
import os

# ==========================================
# 1. パス設定 (ユーザー環境に合わせる)
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
INPUT_FILE = DATA_DIR / "collected_news_historical_full.csv"

# 分析結果の保存先
OUTPUT_DIR = DATA_DIR / "analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"--- ニュースデータ分析開始: {INPUT_FILE.name} ---")
    
    if not INPUT_FILE.exists():
        print("エラー: 入力ファイルが見つかりません。")
        return

    # 1. データの読み込み
    try:
        # 巨大ファイルでも読めるように型指定などは最小限に
        df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig', low_memory=False)
    except Exception as e:
        print(f"読み込みエラー: {e}")
        return

    print(f"全データ件数: {len(df):,} 行")

    # 2. 前処理
    # カラム名の統一
    if 'Date' not in df.columns or 'Code' not in df.columns:
        print("必要なカラム(Date, Code)がありません。")
        print(df.columns)
        return

    # 日付型変換
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date']) # 日付パース失敗行は除外
    df['Code'] = df['Code'].astype(str)

    # 3. 分析①: 銘柄別集計 (Code)
    print("\n[1] 銘柄別集計を作成中...")
    stats_by_code = df.groupby('Code').agg(
        Articles=('Title', 'count'),
        Start_Date=('Date', 'min'),
        End_Date=('Date', 'max')
    ).sort_values('Articles', ascending=False)
    
    # 期間(月数)を計算
    stats_by_code['Months'] = ((stats_by_code['End_Date'] - stats_by_code['Start_Date']) / pd.Timedelta(days=30)).astype(int)
    stats_by_code['Articles_Per_Month'] = (stats_by_code['Articles'] / stats_by_code['Months'].replace(0, 1)).round(1)

    # 4. 分析②: キーワード別集計 (Keyword)
    print("[2] キーワード別集計を作成中...")
    stats_by_keyword = df.groupby(['Code', 'Keyword']).agg(
        Articles=('Title', 'count'),
        Start_Date=('Date', 'min'),
        End_Date=('Date', 'max')
    ).sort_values(['Code', 'Articles'], ascending=[True, False])

    # 5. 分析③: 時系列推移 (月別件数)
    print("[3] 月別ヒートマップ用データを作成中...")
    df['YearMonth'] = df['Date'].dt.to_period('M')
    
    # 行: Code, 列: YearMonth, 値: 件数
    monthly_matrix = df.pivot_table(
        index='Code', 
        columns='YearMonth', 
        values='Title', 
        aggfunc='count', 
        fill_value=0
    )

    # 6. 結果の保存と表示
    
    # (1) 銘柄別サマリー保存
    file_code = OUTPUT_DIR / "news_stats_by_code.csv"
    stats_by_code.to_csv(file_code, encoding='utf-8-sig')
    
    # (2) キーワード別サマリー保存
    file_keyword = OUTPUT_DIR / "news_stats_by_keyword.csv"
    stats_by_keyword.to_csv(file_keyword, encoding='utf-8-sig')
    
    # (3) 月別推移保存
    file_monthly = OUTPUT_DIR / "news_stats_monthly_matrix.csv"
    monthly_matrix.to_csv(file_monthly, encoding='utf-8-sig')

    print("\n--- 分析結果サマリー ---")
    
    print("\n★ マクロニュース (Code=9999) の状況:")
    if '9999' in stats_by_code.index:
        print(stats_by_code.loc[['9999']])
        print("\n  内訳:")
        try:
            print(stats_by_keyword.loc['9999'])
        except:
            pass
    else:
        print("  マクロニュースがありません。")

    print("\n★ 個別銘柄トップ5 (件数順):")
    # 9999以外を表示
    print(stats_by_code[stats_by_code.index != '9999'].head(5))

    print("\n★ データが少ない銘柄ワースト5:")
    print(stats_by_code.tail(5))

    print(f"\n完了しました。詳細はフォルダを確認してください:\n{OUTPUT_DIR}")

if __name__ == "__main__":
    main()