import pandas as pd
from pathlib import Path
import os
import glob
import re

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
XBRL_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "03_organized_by_code"

# 分析対象リスト（ここにあるコードのフォルダだけを見に行きます）
TARGET_LIST_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
# TARGET_LIST_FILE = BASE_DIR / "3_reports" / "analysis_30_candidates" / "selected_30_candidates_list.csv"

# 出力ファイル
OUTPUT_FILE = DATA_DIR / "extracted_financial_data.csv"

# --- 抽出対象の定義 ---
# キーワード（項目名に含まれていれば抽出）
TARGET_KEYWORDS = [
    "売上", "収益", "営業利益", "経常利益", "税引前利益", "当期純利益", "包括利益", 
    "資産", "負債", "純資産", "現金"
]

# 優先的に拾う要素ID（タグ分析から特定した主要タグ）
PRIORITY_TAGS = [
    # 日本基準
    "jppfs_cor:NetSales",             # 売上高
    "jppfs_cor:OperatingIncome",      # 営業利益
    "jppfs_cor:OrdinaryIncome",       # 経常利益
    "jppfs_cor:ProfitLossAttributableToOwnersOfParent", # 親会社株主に帰属する当期純利益
    
    # IFRS
    "jpigp_cor:RevenueIFRS",          # 売上収益
    "jpigp_cor:OperatingProfitLossIFRS", # 営業利益
    "jpigp_cor:ProfitLossAttributableToOwnersOfParentIFRS", # 親会社の所有者に帰属する当期利益
    
    # 経営指標（サマリー）
    "jpcrp_cor:NetSalesSummaryOfBusinessResults",
    "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults",
    "jpcrp_cor:OperatingIncomeSummaryOfBusinessResults"
]

# ==========================================
# 2. 関数定義
# ==========================================
def get_target_codes(file_path):
    """分析対象の銘柄コードリストを読み込む"""
    if not file_path.exists():
        print(f"Warning: Target list file not found: {file_path}")
        return []
    
    try:
        df = pd.read_csv(file_path)
        # カラム名の揺らぎ吸収 ('code', 'Code', 'SC'など)
        for col in df.columns:
            if col.lower() in ['code', 'sc', 'security_code']:
                codes = df[col].astype(str).unique()
                print(f"Target companies loaded: {len(codes)} codes.")
                return codes
        print("Error: Could not find 'code' column in target list.")
        return []
    except Exception as e:
        print(f"Error loading target list: {e}")
        return []

def read_xbrl_csv(path):
    """CSVをロバストに読み込む（区切り文字・エンコーディング自動判定）"""
    # 優先順位: タブ区切り(UTF-16) -> タブ区切り(UTF-8) -> カンマ区切り
    patterns = [
        ('\t', 'utf-16'),
        ('\t', 'utf-8'),
        (',', 'utf-8'),
        ('\t', 'cp932')
    ]
    
    for sep, enc in patterns:
        try:
            df = pd.read_csv(path, sep=sep, encoding=enc)
            # カラム名の正規化（空白削除、引用符削除）
            df.columns = [c.strip().replace('"', '') for c in df.columns]
            
            # 必須カラムチェック
            if '項目名' in df.columns and '値' in df.columns:
                return df
        except Exception:
            continue
    return None

def is_target_row(row):
    """その行が抽出対象か判定する"""
    item_name = str(row.get('項目名', ''))
    element_id = str(row.get('要素ID', ''))
    
    # 1. 優先タグ（ID）と完全一致するか？
    if element_id in PRIORITY_TAGS:
        return True
    
    # 2. キーワードが項目名に含まれるか？
    for kw in TARGET_KEYWORDS:
        if kw in item_name:
            return True
            
    return False

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    print("--- Financial Data Extraction Start ---")
    
    # 1. 対象コード取得
    target_codes = get_target_codes(TARGET_LIST_FILE)
    if len(target_codes) == 0:
        print("No target codes found. Exiting.")
        return

    all_data = []
    processed_files = 0
    
    # 2. フォルダ走査
    for code in target_codes:
        target_dir = XBRL_DIR / code
        
        if not target_dir.exists():
            # print(f"Skipping {code}: Folder does not exist.")
            continue
            
        # CSVファイル取得
        csv_files = list(target_dir.glob("*.csv"))
        if not csv_files:
            continue
            
        print(f"Processing {code} ({len(csv_files)} files)...")
        
        for csv_file in csv_files:
            df = read_xbrl_csv(csv_file)
            if df is None:
                continue
                
            # 3. データの抽出
            for _, row in df.iterrows():
                if is_target_row(row):
                    # 数値のクリーニング
                    val_str = str(row.get('値', ''))
                    # 数値変換可能なものだけ残す（空文字やテキストは除外したいが、注記などは残る可能性あり）
                    
                    all_data.append({
                        'Code': code,
                        'File': csv_file.name,
                        'ItemName': row.get('項目名'),
                        'ElementID': row.get('要素ID'),
                        'Value': val_str,
                        'Unit': row.get('単位'),
                        'Period': row.get('期間・時点'),
                        'ContextID': row.get('コンテキストID')
                    })
            
            processed_files += 1

    # 4. 保存
    if all_data:
        df_result = pd.DataFrame(all_data)
        
        # 重複削除（同じファイルから同じ項目が複数行出る場合があるため）
        df_result = df_result.drop_duplicates()
        
        print("\nExtraction complete!")
        print(f"Processed Files: {processed_files}")
        print(f"Extracted Rows: {len(df_result)}")
        print(df_result.head())
        
        df_result.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Saved to: {OUTPUT_FILE}")
    else:
        print("No matching financial data found.")

if __name__ == "__main__":
    main()