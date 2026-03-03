import pandas as pd
import re
from pathlib import Path

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
EDINET_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"

# テスト対象ファイル
META_FILE = DATA_DIR / "integrated_financial_and_stock_data_v2.csv"
FINANCIAL_FILE = DATA_DIR / "edinet_features_financials_hybrid.csv"
STOCK_FILE = DATA_DIR / "stock_data_features_v1.csv"

def main():
    print("--- 結合ロジック検証テスト (Linkage Test) ---")

    # 1. 必須ファイルの存在確認
    if not META_FILE.exists():
        print(f"❌ 致命的エラー: メタデータ({META_FILE.name})がありません。これがないと結合できません。")
        return
    if not FINANCIAL_FILE.exists():
        print(f"❌ エラー: 財務データ({FINANCIAL_FILE.name})がありません。")
        return
    if not EDINET_DIR.exists():
        print(f"❌ エラー: EDINETフォルダ({EDINET_DIR.name})が見つかりません。DocIDの日付特定に必要です。")
        return

    # 2. データのロード
    print("データを読み込み中...")
    df_meta = pd.read_csv(META_FILE, low_memory=False)
    df_fin = pd.read_csv(FINANCIAL_FILE, low_memory=False)
    df_stock = pd.read_csv(STOCK_FILE, usecols=['Date', 'Code'], low_memory=False) # 必要な列だけ

    print(f"  - 財務データ件数 (DocIDベース): {len(df_fin):,} 件")
    print(f"  - メタデータ件数: {len(df_meta):,} 件")

    # 3. DocID -> PeriodEnd のマッピング作成 (実際の結合ロジックを再現)
    print("\n[Step 1] DocID -> PeriodEnd (決算期末日) の特定")
    docid_to_period = {}
    folders = [p for p in EDINET_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
    
    # サンプリングではなく全件チェック (高速化版)
    count_found = 0
    for f in folders:
        # ファイル名から日付を抜く
        # 例: ..._2020-06-30_... -> 2020-06-30
        try:
            # フォルダ内のCSV名を取得 (globは重いので、os.scandir的な処理の方が速いが、今回はシンプルに)
            # XBRL_TO_CSVがあればそこを見る
            target_dir = f / "XBRL_TO_CSV"
            if target_dir.exists():
                files = list(target_dir.glob("*.csv"))
            else:
                files = list(f.glob("*.csv"))
            
            if files:
                # ファイル名の中から日付っぽいものを正規表現で探す
                m = re.search(r'_(\d{4}-\d{2}-\d{2})_', files[0].name)
                if m:
                    docid_to_period[f.name] = m.group(1)
                    count_found += 1
        except Exception:
            continue
            
    print(f"  -> 日付特定成功: {count_found} / {len(folders)} フォルダ")
    
    if count_found == 0:
        print("❌ 警告: DocIDから日付が一つも特定できませんでした。フォルダ構造か正規表現が間違っています。")
        return

    # 4. ブリッジテーブル作成
    df_bridge = pd.DataFrame(list(docid_to_period.items()), columns=['DocID', 'periodEnd_str'])
    df_bridge['periodEnd'] = pd.to_datetime(df_bridge['periodEnd_str'])
    
    # 5. 結合テスト: Financial -> Bridge -> Meta -> Code
    print("\n[Step 2] 財務データ -> 銘柄コード(Code) の紐付け")
    
    # まず財務データとブリッジを結合
    df_step1 = pd.merge(df_fin, df_bridge, on='DocID', how='inner')
    print(f"  -> 日付特定できた財務データ: {len(df_step1):,} 件")

    # 次にメタデータと結合してCodeを取得
    # メタデータ側の準備
    df_meta['periodEnd'] = pd.to_datetime(df_meta['periodEnd'])
    if 'Code' in df_meta.columns:
        df_meta.rename(columns={'Code': 'code'}, inplace=True)
    df_meta['code'] = df_meta['code'].astype(str)
    
    # 重複排除 (Code, periodEndのペアを一意に)
    df_meta_unique = df_meta[['code', 'periodEnd', 'submitDateTime']].drop_duplicates()
    
    # 結合
    df_linked = pd.merge(df_step1, df_meta_unique, on='periodEnd', how='inner')
    
    success_rate = len(df_linked) / len(df_fin)
    print(f"  -> 銘柄コード特定成功: {len(df_linked):,} 件 (カバー率: {success_rate:.1%})")

    if success_rate < 0.5:
        print("⚠️ 警告: 紐付け成功率が低すぎます (50%未満)。メタデータの期間とDocIDの期間がズレている可能性があります。")
    
    # 6. 最終テスト: 株価データへの着地
    print("\n[Step 3] 株価データへの結合シミュレーション")
    
    # submitDateTime (発表日) を Date に変換
    df_linked['Date'] = pd.to_datetime(df_linked['submitDateTime']).dt.normalize()
    
    # 株価データ側の準備
    if 'Code' in df_stock.columns:
        df_stock.rename(columns={'Code': 'code'}, inplace=True)
    df_stock['code'] = df_stock['code'].astype(str)
    df_stock['Date'] = pd.to_datetime(df_stock['Date'])
    
    # 結合キー (code, Date) が株価データに存在するかチェック
    # mergeだと重いので、setでチェック
    stock_keys = set(zip(df_stock['code'], df_stock['Date']))
    linked_keys = set(zip(df_linked['code'], df_linked['Date']))
    
    valid_keys = linked_keys.intersection(stock_keys)
    
    print(f"  -> 発表日が株価データ範囲内にある件数: {len(valid_keys):,} 件")
    
    # 結論
    print("\n" + "="*40)
    print("   判定結果")
    print("="*40)
    
    if len(valid_keys) > 1000:
        print("✅ OK: データ結合は正常に機能します。")
        print("       安心して merge_features_final_v4.py を実行してください。")
    else:
        print("❌ NG: ほとんどのデータが結合できません。")
        print("       考えられる原因:")
        print("       1. integrated_financial_and_stock_data_v2.csv が古い、または空")
        print("       2. DocIDと決算日のマッピングに失敗している")
        print("       3. 株価データの日付範囲と決算発表日が重なっていない")

if __name__ == "__main__":
    main()