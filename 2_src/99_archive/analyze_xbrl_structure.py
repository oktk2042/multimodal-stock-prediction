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

# 調査するサンプル数 (全件やると遅いので、ランダムに500社ほど見る)
SAMPLE_SIZE = 11057

# 探したいキーワード (タグ名に含まれているかチェック)
SEARCH_KEYWORDS = ["Sales", "Revenue", "OperatingIncome", "OperatingProfit", "OrdinaryIncome"]

# ==========================================
# ユーティリティ
# ==========================================
def read_csv_robust(path):
    """どんなCSVでも意地でも読む関数"""
    encodings = ['utf-8', 'utf-16', 'cp932', 'shift_jis']
    separators = ['\t', ',']
    
    for enc in encodings:
        for sep in separators:
            try:
                df = pd.read_csv(path, sep=sep, encoding=enc)
                # 少なくとも2列以上ないとCSVとして怪しい
                if df.shape[1] > 1:
                    return df
            except:
                continue
    return pd.DataFrame()

def main():
    print("--- XBRLデータ構造分析（サンプリング調査）開始 ---")
    
    # フォルダ一覧取得
    doc_folders = [p for p in INPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
    print(f"全フォルダ数: {len(doc_folders)}")
    
    # ランダムサンプリング
    if len(doc_folders) > SAMPLE_SIZE:
        target_folders = random.sample(doc_folders, SAMPLE_SIZE)
    else:
        target_folders = doc_folders
        
    print(f"調査対象: {len(target_folders)} フォルダ (ランダム抽出)")

    # 集計用カウンタ
    column_names_counter = Counter()
    sales_tags_counter = Counter()
    profit_tags_counter = Counter()
    context_counter = Counter()
    
    for folder in tqdm(target_folders):
        # XBRL_TO_CSV フォルダを探す
        target_dir = folder / "XBRL_TO_CSV"
        if target_dir.exists():
            csv_files = list(target_dir.glob("*.csv"))
        else:
            csv_files = list(folder.rglob("*.csv"))
            
        # 報告書本体っぽいCSVを探す (manifestなどは除外したい)
        # 通常、ファイルサイズが大きいものや、名前に 'jpcrp' などが入る
        main_csv = None
        for f in csv_files:
            if "jpcrp" in f.name or "jpaud" in f.name:
                main_csv = f
                break
        if not main_csv and csv_files:
            main_csv = csv_files[0]
            
        if not main_csv:
            continue
            
        # 読み込み
        df = read_csv_robust(main_csv)
        if df.empty:
            continue
        
        # 1. 列名の調査
        cols = list(df.columns)
        # 組み合わせとしてカウント (文字列化)
        column_names_counter[str(sorted(cols))] += 1
        
        # ID列、コンテキスト列を特定
        id_col = next((c for c in cols if "要素ID" in c or "ElementID" in c or "name" == c), None)
        ctx_col = next((c for c in cols if "コンテキストID" in c or "ContextID" in c), None)
        
        if not id_col:
            continue
        
        # 2. タグの出現頻度調査
        # 売上系
        sales_rows = df[df[id_col].astype(str).str.contains("Sales|Revenue", case=False, na=False)]
        for tag in sales_rows[id_col].unique():
            # 短すぎる、長すぎるゴミを除去してカウント
            if 5 < len(str(tag)) < 1000:
                sales_tags_counter[tag] += 1
                
        # 利益系
        profit_rows = df[df[id_col].astype(str).str.contains("OperatingIncome|OperatingProfit", case=False, na=False)]
        for tag in profit_rows[id_col].unique():
            if 5 < len(str(tag)) < 1000:
                profit_tags_counter[tag] += 1
                
        # 3. コンテキストの調査
        if ctx_col:
            # よく使われるコンテキストID上位を記録
            contexts = df[ctx_col].dropna().astype(str).tolist()
            # 各ファイルで使われているコンテキストのセットをカウント
            unique_ctx = set(contexts)
            for ctx in unique_ctx:
                context_counter[ctx] += 1

    # === 結果表示 ===
    print("\n" + "="*50)
    print("分析結果レポート")
    print("="*50)
    
    print("\n【1. 列名のパターン (Top 50)】")
    for cols, count in column_names_counter.most_common(50):
        print(f"{count}件: {cols}")

    print("\n【2. 売上高に関連するタグ (Top 100)】")
    print("※ここに出ているタグを抽出リストに加えるべきです")
    for tag, count in sales_tags_counter.most_common(100):
        print(f"{count}件: {tag}")

    print("\n【3. 営業利益に関連するタグ (Top 100)】")
    for tag, count in profit_tags_counter.most_common(100):
        print(f"{count}件: {tag}")
        
    print("\n【4. 期間コンテキストID (Top 100)】")
    print("※CurrentYear, CurrentYTD などが重要")
    for ctx, count in context_counter.most_common(100):
        print(f"{count}件: {ctx}")

if __name__ == "__main__":
    main()