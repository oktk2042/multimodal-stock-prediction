import pandas as pd
from pathlib import Path
import json

def analyze_all_stock_data_coverage():
    """
    master_stock_list.csv に基づいて、保有する全銘柄の株価データの
    期間、連続性、欠損状況を調査し、サマリーレポートを出力する。
    """
    try:
        # --- 1. ファイルパスの設定 ---
        base_path = Path("C:/M2_Research_Project/1_data")
        processed_path = base_path / "processed"

        # 入力ファイル
        stock_features_file = processed_path / "stock_data_features_v1.csv"
        # [修正] 200銘柄リストの代わりに、全銘柄マスターリストを使用
        master_list_file = processed_path / "master_stock_list.csv"

        # 出力ファイル
        output_file = processed_path / "data_coverage_summary_ALL_STOCKS.csv"

        print("--- 全銘柄のデータカバレッジ分析を開始します ---")

        # --- 2. 分析対象の全銘柄リストを読み込み ---
        print(f"'{master_list_file.name}' から全銘柄リストを読み込んでいます...")
        # [修正] master_stock_list.csv を読み込み、'code'列を文字列型のリストとして取得
        master_df = pd.read_csv(master_list_file, dtype={'code': str})
        target_codes = master_df['code'].tolist()
        print(f"対象となる全銘柄数: {len(target_codes)}")
        
        # --- 3. 日次株価データを読み込み ---
        print(f"'{stock_features_file.name}' を読み込んでいます... (時間がかかる場合があります)")
        stock_df = pd.read_csv(
            stock_features_file,
            parse_dates=['Date'],
            dtype={'Code': str}
        )
        print("読み込み完了。")
        
        # --- 4. 全銘柄のデータ状況を分析 ---
        analysis_results = []
        print("各銘柄のデータ状況を分析中...")

        # 処理の進捗を表示するために tqdm を使う（オプション）
        # pip install tqdm が必要
        try:
            from tqdm import tqdm
            iterator = tqdm(target_codes)
        except ImportError:
            iterator = target_codes

        for code in iterator:
            ticker_df = stock_df[stock_df['Code'] == code].copy()

            if ticker_df.empty:
                analysis_results.append({
                    'Code': code,
                    'データ有無': '無し',
                    'データ開始日': None,
                    'データ終了日': None,
                    'レコード数': 0,
                    '営業日カバレッジ (%)': 0,
                    '欠損項目': 'N/A'
                })
                continue

            start_date = ticker_df['Date'].min()
            end_date = ticker_df['Date'].max()
            record_count = len(ticker_df)

            # データの連続性を評価 (営業日ベース)
            expected_days = pd.bdate_range(start=start_date, end=end_date)
            coverage_percentage = (record_count / len(expected_days)) * 100 if len(expected_days) > 0 else 0

            # 各列の欠損数をチェック
            missing_counts = ticker_df.isnull().sum()
            missing_items = missing_counts[missing_counts > 0].to_dict()

            analysis_results.append({
                'Code': code,
                'データ有無': '有り',
                'データ開始日': start_date.strftime('%Y-%m-%d'),
                'データ終了日': end_date.strftime('%Y-%m-%d'),
                'レコード数': record_count,
                '営業日カバレッジ (%)': round(coverage_percentage, 2),
                '欠損項目': json.dumps(missing_items, ensure_ascii=False) if missing_items else '無し'
            })
            
        # --- 5. 結果をDataFrameに変換して保存 ---
        analysis_df = pd.DataFrame(analysis_results)
        # 銘柄マスターの企業名も結合して、レポートを分かりやすくする
        analysis_df = pd.merge(master_df[['code', 'name']], analysis_df, left_on='code', right_on='Code', how='left')
        analysis_df.drop(columns=['code'], inplace=True)
        
        analysis_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n--- 分析完了 ---")
        print(f"分析結果が '{output_file}' に保存されました。")

        return analysis_df

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None

if __name__ == '__main__':
    summary_df = analyze_all_stock_data_coverage()
    if summary_df is not None:
        print("\n--- 分析サマリー（一部抜粋） ---")
        print("▼ 営業日カバレッジが低い銘柄 (ワースト5)")
        print(summary_df.sort_values('営業日カバレッジ (%)').head())
        print("\n▼ データが存在しない銘柄 (5件)")
        print(summary_df[summary_df['データ有無'] == '無し'].head())