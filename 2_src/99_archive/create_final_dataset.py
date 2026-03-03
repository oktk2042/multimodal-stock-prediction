import pandas as pd
from pathlib import Path

def create_final_analysis_dataset():
    """
    前処理済みの日次株価データと、統合済みの財務データを結合し、
    Transformerモデルへの入力に適した最終データセットを作成する。
    """
    try:
        # --- 1. ファイルパスの設定 ---
        base_path = Path("C:/M2_Research_Project/1_data")
        processed_path = base_path / "processed"

        # 入力ファイル
        daily_data_file = processed_path / "preprocessed_top200_daily_data.csv"
        financial_data_file = processed_path / "integrated_final_dataset_v3.csv"

        # 出力ファイル
        output_file = processed_path / "final_model_input_dataset.csv"

        print("--- 最終データセットの作成を開始します ---")

        # --- 2. 2つの主要データを読み込み ---
        print("日次株価データを読み込んでいます...")
        daily_df = pd.read_csv(
            daily_data_file,
            parse_dates=['Date'],
            dtype={'Code': str}
        )

        print("財務データを読み込んでいます...")
        financial_df = pd.read_csv(
            financial_data_file,
            parse_dates=['periodEnd', 'Date'], # 財務データ側のDateも念のためパース
            dtype={'code': str}
        )

        # --- 3. 財務データから必要な列を選択 ---
        # 結合に必要なキーと、追加したい財務情報のみに絞り込む
        financial_subset_df = financial_df[[
            'code',
            'Date', # 結合キーとして使用 (決算発表後の株価営業日)
            'NetSales',
            'OperatingIncome',
            'NetIncome',
            'TotalAssets',
            'NetAssets'
        ]].rename(columns={'code': 'Code'}) # キー列の名前を合わせる

        # 同じ日付に複数の財務報告がある場合（稀なケース）、最新のものを残す
        financial_subset_df = financial_subset_df.drop_duplicates(subset=['Code', 'Date'], keep='last')
        
        print("日次データと財務データを結合します...")
        
        # --- 4. 日次データを主軸に、財務データを左結合 ---
        # daily_dfの全行は維持し、日付とコードが一致する行に財務データを付与
        merged_df = pd.merge(
            daily_df,
            financial_subset_df,
            on=['Date', 'Code'],
            how='left'
        )
        
        print("財務情報をフォワードフィル（前方補完）します...")

        # --- 5. 財務データのフォワードフィル ---
        # 銘柄ごとにグループ化し、NaNを前の期の数値で埋める
        financial_columns = ['NetSales', 'OperatingIncome', 'NetIncome', 'TotalAssets', 'NetAssets']
        merged_df[financial_columns] = merged_df.groupby('Code')[financial_columns].transform(
            lambda x: x.ffill()
        )

        # ffill後も残るNaNは、その銘柄の最初の決算データがない期間なので、0で埋めるか、行を削除する
        # ここでは0で埋める（fillna(0)）を選択
        merged_df[financial_columns] = merged_df[financial_columns].fillna(0)

        # --- 6. 結果をCSVファイルに保存 ---
        merged_df.sort_values(by=['Code', 'Date'], inplace=True)
        merged_df.to_csv(output_file, index=False, encoding='utf-8-sig')

        print(f"\n--- 処理完了 ---")
        print(f"最終的な分析用データセットが '{output_file}' に保存されました。")
        
        return merged_df

    except FileNotFoundError as e:
        print(f"エラー: ファイルが見つかりませんでした。パスを確認してください: {e.filename}")
        return None
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None

if __name__ == '__main__':
    final_dataset = create_final_analysis_dataset()
    if final_dataset is not None:
        print("\n--- 最終データセットのサンプル（ソフトバンクグループ: 9984）---")
        # 決算発表があったあたりでデータがどうなっているか確認
        sbg_sample = final_dataset[final_dataset['Code'] == '9984']
        print(sbg_sample[sbg_sample['Date'].dt.year == 2024].head(10))
