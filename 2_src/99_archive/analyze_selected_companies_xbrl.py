import pandas as pd
from pathlib import Path
import os
import glob

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
XBRL_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "03_organized_by_code"

# 分析対象の銘柄リストを取得するファイル
# 全200銘柄を対象にする場合
TARGET_LIST_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"
# または、選定した30銘柄のみを対象にする場合
# TARGET_LIST_FILE = PROJECT_ROOT / "3_reports" / "analysis_30_candidates" / "selected_30_candidates_list.csv"

# 分析結果の保存先
OUTPUT_FILE = DATA_DIR / "xbrl_analysis_results.csv"

# 抽出したい財務項目のタグ（正規表現でマッチング）
# ※日本語ラベル（項目名）で探すのが確実です
TARGET_KEYWORDS = ["売上高", "売上収益", "営業利益", "経常利益", "当期純利益"]

# ==========================================
# 2. 関数定義
# ==========================================
def get_target_codes(file_path):
    """分析対象の銘柄コードリストを取得する"""
    print(f"Loading target codes from: {file_path}")
    try:
        df = pd.read_csv(file_path)
        # カラム名の揺らぎ吸収
        col_name = 'code' if 'code' in df.columns else 'Code'
        codes = df[col_name].unique().astype(str)
        print(f"Target companies found: {len(codes)}")
        return codes
    except Exception as e:
        print(f"Error loading target list: {e}")
        return []

def read_xbrl_csv_robust(path):
    """エンコーディングや区切り文字を自動判定してCSVを読む"""
    # XBRL_TO_CSVの出力は通常タブ区切り(tsv)で、エンコーディングはutf-16が多いが、
    # ツールによってはutf-8やcp932の場合もあるため総当たりする
    encodings = ['utf-16', 'utf-8', 'cp932', 'shift_jis']
    separators = ['\t', ',']
    
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                # 正常に読めたかの簡易チェック（項目名カラムがあるか）
                if '項目名' in df.columns or 'ElementLabel' in df.columns:
                    return df
            except Exception:
                continue
    return None

def analyze_company_folder(code_dir, code):
    """1つの銘柄フォルダ内の全CSVを走査して数値を抽出する"""
    results = []
    
    # フォルダ内のCSVファイルを検索
    csv_files = list(code_dir.glob("*.csv"))
    
    for csv_file in csv_files:
        df = read_xbrl_csv_robust(csv_file)
        if df is None:
            continue
            
        # 必要なカラムがあるか確認 ('項目名', '値', '期間・時点' など)
        # ツールによってカラム名が違う場合があるので調整
        # 例: Arelleの出力形式などを想定
        
        # 一般的なカラム名の正規化
        df.columns = [c.strip() for c in df.columns]
        
        if '項目名' not in df.columns or '値' not in df.columns:
            continue
            
        # ターゲット項目を検索
        for keyword in TARGET_KEYWORDS:
            # 項目名にキーワードが含まれる行を抽出
            mask = df['項目名'].astype(str).str.contains(keyword, na=False)
            found_rows = df[mask]
            
            for _, row in found_rows.iterrows():
                # 値が数値として有効かチェック
                val_str = str(row['値'])
                if not val_str.replace('.','').replace('-','').isdigit():
                    continue
                    
                # 期間情報の取得（いつのデータか）
                period = row.get('期間・時点', 'Unknown')
                if 'CurrentYear' in str(row.get('コンテキストID', '')):
                    period_type = 'Current'
                else:
                    period_type = 'Other'

                results.append({
                    'Code': code,
                    'File': csv_file.name,
                    'Item': row['項目名'],
                    'Value': val_str,
                    'Period': period,
                    'Context': row.get('コンテキストID', '')
                })
                
    return results

# ==========================================
# 3. メイン処理
# ==========================================
def main():
    # 1. 対象銘柄コードの取得
    target_codes = get_target_codes(TARGET_LIST_FILE)
    if len(target_codes) == 0:
        return

    all_results = []
    
    print("\n--- Starting Analysis ---")
    
    # 2. 各銘柄フォルダを処理
    for code in target_codes:
        code_dir = XBRL_DIR / code
        
        if not code_dir.exists():
            # print(f"Skipping {code}: Folder not found in organized directory.")
            continue
            
        print(f"Processing {code}...")
        company_results = analyze_company_folder(code_dir, code)
        all_results.extend(company_results)
        
    # 3. 結果の保存
    if all_results:
        df_results = pd.DataFrame(all_results)
        print(f"\nAnalysis complete. Found {len(df_results)} data points.")
        print(df_results.head())
        
        df_results.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
        print(f"Results saved to: {OUTPUT_FILE}")
    else:
        print("No financial data found matching the keywords.")

if __name__ == "__main__":
    main()