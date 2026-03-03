import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"
OUTPUT_FILE = PROJECT_ROOT / "1_data" / "processed" / "edinet_features_financials_rulebased.csv"

# 抽出したい項目のキーワード（正規表現）
# 優先順位: IFRS -> 日本基準 の順で探すためのリスト
TARGET_KEYS = {
    "NetSales": [
        "RevenueIFRSSummaryOfBusinessResults", # IFRS 売上収益
        "NetSalesSummaryOfBusinessResults",    # 日本基準 売上高
        "OperatingRevenue1SummaryOfBusinessResults", # 営業収益
    ],
    "OperatingIncome": [
        "OperatingProfitLossIFRSSummaryOfBusinessResults", # IFRS 営業利益
        "OperatingIncomeLossSummaryOfBusinessResults",     # 日本基準 営業利益
        "OrdinaryIncomeLossSummaryOfBusinessResults"       # 経常利益 (営業利益がない場合の代用)
    ]
}

def read_csv_robust(path):
    # タブ区切り(TSV)やエンコーディングに対応
    for enc in ['utf-8', 'utf-16', 'cp932']:
        try:
            return pd.read_csv(path, sep='\t', encoding=enc)
        except:
            try:
                return pd.read_csv(path, sep=',', encoding=enc)
            except:
                continue
    return pd.DataFrame()

def extract_from_folder(folder_path):
    # XBRL_TO_CSVフォルダ内のCSVを走査
    target_dir = folder_path / "XBRL_TO_CSV"
    if not target_dir.exists():
        csv_files = list(folder_path.rglob("*.csv"))
    else:
        csv_files = list(target_dir.glob("*.csv"))
    
    extracted = {"NetSales": None, "OperatingIncome": None}
    
    for csv_file in csv_files:
        df = read_csv_robust(csv_file)
        if df.empty: continue
        
        # カラム名チェック (要素ID, 値 があるか)
        # 実際のカラム名は "要素ID", "値" など
        id_col = next((c for c in df.columns if "要素ID" in c or "ElementID" in c), None)
        val_col = next((c for c in df.columns if "値" in c or "Value" in c), None)
        ctx_col = next((c for c in df.columns if "コンテキストID" in c or "ContextID" in c), None)
        
        if not id_col or not val_col:
            continue
            
        # 必要な行だけフィルタリング
        # ContextIDが "Current..." (当期) のものを優先したいが、
        # まずは単純にタグで検索し、数値が入っているものを探す
        
        for key, tags in TARGET_KEYS.items():
            if extracted[key] is not None: continue # 既に取得済みならスキップ
            
            for tag in tags:
                # タグを含む行を抽出
                # 例: jpcrp_cor:NetSales...
                mask = df[id_col].astype(str).str.contains(tag, case=False, na=False)
                
                # コンテキストフィルタ (当期データぽいもの)
                if ctx_col:
                    # CurrentYear, CurrentYTD, CurrentQuarter などを優先
                    mask_ctx = df[ctx_col].astype(str).str.contains("Current", case=False, na=False)
                    hits = df[mask & mask_ctx]
                    if hits.empty:
                        hits = df[mask] # コンテキストで絞れなければ全体から
                else:
                    hits = df[mask]
                
                if not hits.empty:
                    # 最初に見つかった数値を採用
                    try:
                        val = hits.iloc[0][val_col]
                        # 数値化 (カンマ削除など)
                        val_num = float(str(val).replace(',', ''))
                        extracted[key] = val_num
                        break # タグが見つかったら次の項目へ
                    except:
                        continue

    return extracted

def main():
    print("--- 財務数値抽出 (ルールベース) 開始 ---")
    
    results = []
    doc_folders = [p for p in INPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
    
    print(f"対象フォルダ数: {len(doc_folders)}")
    
    for folder in tqdm(doc_folders):
        doc_id = folder.name
        data = extract_from_folder(folder)
        
        # 少なくともどちらかが取れていれば保存
        if data["NetSales"] is not None or data["OperatingIncome"] is not None:
            results.append({
                "DocID": doc_id,
                "NetSales_RuleBased": data["NetSales"],
                "OperatingIncome_RuleBased": data["OperatingIncome"]
            })
            
    if results:
        df_res = pd.DataFrame(results)
        df_res.to_csv(OUTPUT_FILE, index=False)
        print(f"\n完了: {OUTPUT_FILE}")
        print(f"抽出件数: {len(df_res)}")
        print(df_res.head())
    else:
        print("抽出データがありませんでした。")

if __name__ == "__main__":
    main()