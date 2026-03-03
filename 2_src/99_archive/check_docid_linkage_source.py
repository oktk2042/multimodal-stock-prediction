import pandas as pd
from pathlib import Path
import re

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data"
PROCESSED_DIR = DATA_DIR / "processed"
EDINET_DIR = DATA_DIR / "edinet_reports" / "02_unzipped_files"

def main():
    print("--- DocID <-> Code 紐付け情報探索 ---")

    # 1. 既存のCSVにヒントがないか探す
    candidates = list(DATA_DIR.rglob("*.csv"))
    print(f"探索対象CSV: {len(candidates)} ファイル")
    
    found_map = False
    
    for csv_path in candidates:
        if "processed" in str(csv_path) and "final" in str(csv_path):
            continue
        
        try:
            # 先頭だけ読んでカラム確認
            df = pd.read_csv(csv_path, nrows=5, encoding='utf-8-sig')
            cols = [c.lower() for c in df.columns]
            
            # docid と code (または ticker) の両方を含んでいるか？
            has_docid = any("docid" in c for c in cols)
            has_code = any("code" in c or "ticker" in c or "sec_code" in c for c in cols)
            
            if has_docid and has_code:
                print(f"✅ 有力候補発見: {csv_path.name}")
                print(f"   カラム: {list(df.columns)}")
                found_map = True
        except Exception:
            continue

    if not found_map:
        print("\n❌ DocIDとCodeが両方入っているCSVが見つかりませんでした。")
        print("   -> XBRLファイルの中身からEDINETコード(E*****)を特定し、")
        print("      それを証券コードに変換するロジックが必要です。")
        
        # 試しに1つXBRLを開いてEDINETコードがあるか確認
        print("\n[検証] XBRLファイル内のEDINETコード確認")
        folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
        if folders:
            target = folders[0]
            # csvを探す
            csvs = list(target.glob("XBRL_TO_CSV/*.csv"))
            if not csvs:
                csvs = list(target.glob("*.csv"))
            
            if csvs:
                # ファイル名に E***** が含まれていることが多い
                # 例: jpcrp030000-asr-001_E01234-000...
                print(f"サンプルファイル: {csvs[0].name}")
                m = re.search(r'_(E\d{5})', csvs[0].name)
                if m:
                    print(f"   -> EDINETコード発見: {m.group(1)}")
                    print("   -> これを使えば復元可能です！")
                else:
                    print("   -> ファイル名からEDINETコードが見つかりません。中身を検索します。")

if __name__ == "__main__":
    main()