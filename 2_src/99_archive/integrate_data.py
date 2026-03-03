import pandas as pd
import json
from pathlib import Path
import glob
import os

def extract_flexible_financial_value(df, element_ids, context_ids):
    """
    複数の候補となる要素IDとコンテキストIDを試し、最初に見つかった値を返す関数。
    """
    for ctx in context_ids:
        for eid in element_ids:
            value_df = df[(df['要素ID'] == eid) & (df['コンテキストID'] == ctx)]
            if not value_df.empty:
                return pd.to_numeric(value_df['値'].iloc[0], errors='coerce')
    return None

def integrate_full_data_from_source():
    """
    EDINETサマリーを元に、個別の財務報告書CSVから直接データを抽出し、株価データと統合する。
    文字コードと項目名の多様性に対応。
    """
    try:
        # --- 1. ファイルパスと項目名の定義 ---
        base_path = Path("C:/M2_Research_Project/1_data")
        raw_path = base_path / "raw"
        processed_path = base_path / "processed"
        edinet_path = base_path / "edinet_reports"
        unzipped_path = edinet_path / "02_unzipped_files"

        output_file = processed_path / "integrated_final_dataset_v3.csv"

        # 財務項目の候補リストを定義
        NET_SALES_IDS = ["jppfs_cor:NetSales", "ifrs-full_Revenue", "jpcrp_cor:Sales"]
        OP_INCOME_IDS = ["jppfs_cor:OperatingIncome", "ifrs-full_ProfitLossFromOperatingActivities", "jpcrp_cor:OperatingIncome"]
        NET_INCOME_IDS = ["jppfs_cor:ProfitLoss", "ifrs-full_ProfitLoss", "jpcrp_cor:NetIncome"]
        ASSETS_IDS = ["jppfs_cor:Assets", "ifrs-full_Assets"]
        NET_ASSETS_IDS = ["jppfs_cor:NetAssets", "ifrs-full_Equity"]

        # コンテキストIDの候補リストを定義
        YTD_CONTEXTS = ["CurrentYTDDuration_ConsolidatedMember", "CurrentYTDDuration_NonConsolidatedMember"]
        INSTANT_CONTEXTS = ["CurrentInstant_ConsolidatedMember", "CurrentInstant_NonConsolidatedMember"]

        print("--- ファイル読み込み開始 ---")
        
        # --- 2. マスターデータ等の読み込み ---
        ticker_file = raw_path / "top_200_trad_val_tickers_filtered.txt"
        with open(ticker_file, 'r', encoding='utf-8') as f:
            tickers = [line.strip() for line in f]
        top_200_df = pd.DataFrame(tickers, columns=['TICKER'])
        top_200_df['code'] = top_200_df['TICKER'].str.replace('.T', '', regex=False)

        master_df = pd.read_csv(processed_path / "master_stock_list.csv", dtype={'code': str})
        stock_df = pd.read_csv(processed_path / "stock_data_features_v1.csv", parse_dates=['Date'], dtype={'Code': str})
        stock_df = stock_df.rename(columns={'Code': 'code'})

        summary_df = pd.read_csv(edinet_path / "EDINET_Summary_v3.csv", on_bad_lines='skip', low_memory=False, 
                                 dtype={'secCode': str, 'docID': str})
        summary_df['code'] = summary_df['secCode'].str.slice(0, 4)
        summary_df['periodEnd'] = pd.to_datetime(summary_df['periodEnd'], errors='coerce')
        summary_df['submitDateTime'] = pd.to_datetime(summary_df['submitDateTime'], errors='coerce')
        summary_df.dropna(subset=['code', 'periodEnd', 'docID'], inplace=True)
        
        target_summary_df = summary_df[summary_df['code'].isin(top_200_df['code'])].copy()
        target_summary_df.sort_values(by=['code', 'periodEnd', 'submitDateTime'], ascending=[True, False, False], inplace=True)
        target_summary_df.drop_duplicates(subset=['code', 'periodEnd'], keep='first', inplace=True)

        print("--- 個別財務報告書からのデータ抽出開始 ---")
        financial_records = []
        for index, row in target_summary_df.iterrows():
            doc_id = row['docID']
            report_folder = unzipped_path / doc_id / "XBRL_TO_CSV"
            
            if report_folder.exists():
                csv_files = glob.glob(os.path.join(report_folder, "jpcrp*.csv"))
                if csv_files:
                    try:
                        try:
                            report_df = pd.read_csv(csv_files[0], delimiter='\t', encoding='utf-16-le')
                        except (UnicodeDecodeError, TypeError):
                            report_df = pd.read_csv(csv_files[0], delimiter='\t', encoding='cp932')
                        
                        record = {
                            'code': row['code'],
                            'periodEnd': row['periodEnd'],
                            'docDescription': row['docDescription'],
                            'NetSales': extract_flexible_financial_value(report_df, NET_SALES_IDS, YTD_CONTEXTS),
                            'OperatingIncome': extract_flexible_financial_value(report_df, OP_INCOME_IDS, YTD_CONTEXTS),
                            'NetIncome': extract_flexible_financial_value(report_df, NET_INCOME_IDS, YTD_CONTEXTS),
                            'TotalAssets': extract_flexible_financial_value(report_df, ASSETS_IDS, INSTANT_CONTEXTS),
                            'NetAssets': extract_flexible_financial_value(report_df, NET_ASSETS_IDS, INSTANT_CONTEXTS)
                        }
                        financial_records.append(record)
                    except Exception as e:
                        print(f"  - Warning: ファイル {doc_id} の処理中に予期せぬエラー: {e}")

        financial_df = pd.DataFrame(financial_records)
        print("--- データ抽出完了、最終統合処理へ ---")
        
        final_df = pd.merge(financial_df, master_df, on='code', how='left')
        final_df = pd.merge_asof(
            final_df.sort_values('periodEnd'),
            stock_df.sort_values('Date'),
            left_on='periodEnd',
            right_on='Date',
            by='code',
            direction='forward'
        )

        final_df.dropna(subset=['name'], inplace=True)
        final_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n処理が完了しました。統合データは '{output_file}' に保存されました。")

        return final_df

    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")
        return None

if __name__ == '__main__':
    integrated_df = integrate_full_data_from_source()
    if integrated_df is not None:
        print("\n--- 統合データ（先頭10件） ---")
        display_cols = ['code', 'name', 'periodEnd', 'NetSales', 'OperatingIncome', 'Date', 'Close']
        print(integrated_df[display_cols].head(10).to_markdown(index=False, floatfmt=",.0f"))
        
        print("\n--- ソフトバンクグループ (9984) の時系列データ（一部）---")
        sbg_df = integrated_df[integrated_df['code'] == '9984']
        print(sbg_df[display_cols].to_markdown(index=False, floatfmt=",.0f"))