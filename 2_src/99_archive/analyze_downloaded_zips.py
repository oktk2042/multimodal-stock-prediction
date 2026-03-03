import pandas as pd
from pathlib import Path
import os
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
BASE_DIR = Path("C:/M2_Research_Project/1_data/edinet_reports/")
ZIP_DIR = BASE_DIR / "01_zip_files_indices"
META_FILE = BASE_DIR / "00_metadata" / "metadata_2018_2025_all.csv"
OUTPUT_FILE = Path("C:/M2_Research_Project/1_data/processed/downloaded_data_summary_by_company.csv")

def main():
    print("--- 収集データ状況分析開始 ---")

    # 1. メタデータの読み込み
    if not META_FILE.exists():
        print(f"エラー: メタデータファイルが見つかりません: {META_FILE}")
        return

    print("メタデータを読み込んでいます...")
    try:
        df_meta = pd.read_csv(META_FILE, dtype=str, low_memory=False)
        # secCodeのクリーニング
        df_meta = df_meta.dropna(subset=['secCode'])
        df_meta['secCode'] = df_meta['secCode'].str.replace('.0', '', regex=False)
        df_meta['Code'] = df_meta['secCode'].str[:4]
        df_meta['submitDateTime'] = pd.to_datetime(df_meta['submitDateTime'])
        df_meta['Year'] = df_meta['submitDateTime'].dt.year
    except Exception as e:
        print(f"メタデータ読み込みエラー: {e}")
        return

    # 2. 実在するZIPファイルの確認
    print(f"ZIPフォルダ ({ZIP_DIR}) をスキャン中...")
    if not ZIP_DIR.exists():
        print("エラー: ZIPフォルダが見つかりません。")
        return
        
    downloaded_files = list(ZIP_DIR.glob("*.zip"))
    downloaded_doc_ids = set([f.stem for f in downloaded_files])
    
    print(f"メタデータ上の書類総数: {len(df_meta)}")
    print(f"ダウンロード済みZIP数: {len(downloaded_files)}")
    
    # 3. ダウンロード済みデータのみ抽出
    df_downloaded = df_meta[df_meta['docID'].isin(downloaded_doc_ids)].copy()
    
    if df_downloaded.empty:
        print("警告: ダウンロード済みの書類がメタデータ内に見つかりません（ID不一致の可能性）。")
        return

    print(f"マッチした収集済みデータ数: {len(df_downloaded)}")
    
    # 4. 集計処理
    # (A) 銘柄ごとの集計
    print("銘柄別に集計中...")
    
    # 書類種別の判定関数
    def get_doc_type_label(row):
        code = str(row['docTypeCode'])
        desc = str(row['docDescription'])
        if code == '120':
            return '有報'
        if code == '130':
            return '有報(訂正)'
        if code == '140':
            return '四半期'
        if code == '150':
            return '四半期(訂正)'
        return 'その他'

    df_downloaded['DocTypeLabel'] = df_downloaded.apply(get_doc_type_label, axis=1)
    
    # ピボットテーブル作成: 銘柄 x 年度 の件数
    pivot_year = df_downloaded.pivot_table(
        index=['Code', 'filerName'], 
        columns='Year', 
        values='docID', 
        aggfunc='count', 
        fill_value=0
    )
    
    # ピボットテーブル作成: 銘柄 x 書類種別 の件数
    pivot_type = df_downloaded.pivot_table(
        index=['Code', 'filerName'], 
        columns='DocTypeLabel', 
        values='docID', 
        aggfunc='count', 
        fill_value=0
    )
    
    # 期間情報
    period_stats = df_downloaded.groupby(['Code', 'filerName'])['submitDateTime'].agg(['min', 'max'])
    period_stats.columns = ['FirstSubmission', 'LastSubmission']
    
    # 全結合
    df_summary = pd.concat([pivot_year, pivot_type, period_stats], axis=1).reset_index()
    
    # 総件数
    df_summary['TotalFiles'] = df_summary.apply(lambda x: x[pivot_year.columns].sum(), axis=1)
    
    # カラム整理
    # Yearカラムを文字列にして扱いやすく
    year_cols = [c for c in pivot_year.columns]
    type_cols = [c for c in pivot_type.columns]
    
    base_cols = ['Code', 'filerName', 'TotalFiles', 'FirstSubmission', 'LastSubmission']
    final_cols = base_cols + type_cols + year_cols
    
    # 存在するカラムだけ選ぶ
    final_cols = [c for c in final_cols if c in df_summary.columns]
    df_summary = df_summary[final_cols]
    
    # 保存
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_summary.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
    
    print("\n" + "="*50)
    print("分析結果サマリー")
    print("="*50)
    print(f"収集済み企業数: {df_summary['Code'].nunique()}")
    print(f"総ファイル数: {df_summary['TotalFiles'].sum()}")
    print(f"データ期間: {df_downloaded['Year'].min()}年 〜 {df_downloaded['Year'].max()}年")
    
    print("\n--- 収集数トップ10企業 ---")
    print(df_summary.sort_values('TotalFiles', ascending=False)[['Code', 'filerName', 'TotalFiles']].head(10))
    
    print(f"\n詳細な集計結果を保存しました: {OUTPUT_FILE}")
    print("各銘柄の年度ごとの取得状況（欠けがないか）を確認してください。")

if __name__ == "__main__":
    main()