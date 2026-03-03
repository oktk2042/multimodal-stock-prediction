import pandas as pd
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import random

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

# 調査するサンプル数（フォルダ数）
SAMPLE_SIZE = 11057

# ==========================================
# ユーティリティ
# ==========================================
def read_csv_robust(path):
    for enc in ['utf-8', 'utf-16', 'cp932', 'shift_jis']:
        try:
            return pd.read_csv(path, sep='\t', encoding=enc)
        except:
            try:
                return pd.read_csv(path, sep=',', encoding=enc)
            except:
                continue
    return pd.DataFrame()

def main():
    print("--- XBRL深掘り調査（全ファイルスキャン）開始 ---")
    
    doc_folders = [p for p in INPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
    target_folders = random.sample(doc_folders, min(len(doc_folders), SAMPLE_SIZE))
    
    print(f"調査対象: {len(target_folders)} フォルダ")

    # 発見した情報のカウンタ
    found_file_patterns = Counter() # どのファイル名にデータがあったか
    sales_tags = Counter()          # 売上タグ名
    profit_tags = Counter()         # 利益タグ名
    context_ids = Counter()         # 期間指定のID
    
    hits_count = 0

    for folder in tqdm(target_folders):
        # フォルダ内の全CSVを取得
        target_dir = folder / "XBRL_TO_CSV"
        if target_dir.exists():
            csv_files = list(target_dir.glob("*.csv"))
        else:
            csv_files = list(folder.rglob("*.csv"))
            
        # 全ファイルを走査
        for csv_file in csv_files:
            df = read_csv_robust(csv_file)
            if df.empty: continue
            
            # 列名チェック
            cols = df.columns
            id_col = next((c for c in cols if "要素ID" in c or "ElementID" in c), None)
            val_col = next((c for c in cols if "値" in c or "Value" in c), None)
            ctx_col = next((c for c in cols if "コンテキストID" in c or "ContextID" in c), None)
            
            if not id_col or not val_col:
                continue

            # データの中身を文字列にして検索
            id_series = df[id_col].astype(str)
            
            # ★売上っぽいタグがあるか？
            has_sales = id_series.str.contains("NetSales|Revenue|OperatingRevenue", case=False, na=False).any()
            # ★利益っぽいタグがあるか？
            has_profit = id_series.str.contains("OperatingIncome|OperatingProfit|OrdinaryIncome", case=False, na=False).any()
            
            if has_sales or has_profit:
                # ビンゴ！このファイルが正解です
                
                # 1. ファイル名のパターンを記録 (先頭部分だけ抽出)
                # 例: jpcrp030000-asr-001... -> jpcrp030000
                file_pattern = csv_file.name.split('-')[0]
                found_file_patterns[file_pattern] += 1
                
                # 2. 具体的なタグ名を記録
                if has_sales:
                    tags = df[id_series.str.contains("NetSales|Revenue|OperatingRevenue", case=False, na=False)][id_col].unique()
                    for t in tags:
                        sales_tags[t] += 1
                        
                if has_profit:
                    tags = df[id_series.str.contains("OperatingIncome|OperatingProfit|OrdinaryIncome", case=False, na=False)][id_col].unique()
                    for t in tags:
                        profit_tags[t] += 1

                # 3. コンテキストIDを記録 (CurrentYearDurationなどを探したい)
                if ctx_col:
                    ctxs = df[ctx_col].astype(str).unique()
                    for c in ctxs:
                        # 意味がありそうなコンテキストのみ記録 (長すぎるゴミを除く)
                        if len(c) < 50:
                            context_ids[c] += 1
                
                hits_count += 1
                # 1つのフォルダで1つ見つかれば十分（重複カウントを避けるためbreakしてもいいが、念のため全走査）
                # break 

    print("\n" + "="*50)
    print("深掘り分析結果レポート")
    print("="*50)
    
    if hits_count == 0:
        print("警告: まだ有効なデータファイルが見つかりません。検索キーワードを見直す必要があります。")
    else:
        print(f"\n【1. データが含まれていたファイル名パターン (Top 5)】")
        print("※このファイル名のCSVを優先的に読み込みます")
        for name, count in found_file_patterns.most_common(5):
            print(f"{count}件: {name}")

        print(f"\n【2. 売上高のタグ (Top 50)】")
        print("※これを抽出コードのターゲットにします")
        for tag, count in sales_tags.most_common(50):
            print(f"{count}件: {tag}")

        print(f"\n【3. 営業利益のタグ (Top 50)】")
        for tag, count in profit_tags.most_common(50):
            print(f"{count}件: {tag}")

        print(f"\n【4. コンテキストID (Top 50)】")
        print("※CurrentYearDuration, CurrentYTDDuration などが重要")
        for ctx, count in context_ids.most_common(50):
            print(f"{count}件: {ctx}")

if __name__ == "__main__":
    main()