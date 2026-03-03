from pathlib import Path

import pandas as pd

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力ファイル
TOP200_FILE = DATA_DIR / "top200_stock_list_final.csv"  # 200銘柄リスト
SECTOR_FILE = DATA_DIR / "stock_sector_info.csv"  # セクター情報
NEWS_COUNT_FILE = DATA_DIR / "news_stats_by_code.csv"  # ニュース記事数
NEWS_NAME_FILE = DATA_DIR / "news_stats_by_keyword.csv"  # 企業名(キーワード)


def generate_table():
    print("Generating Appendix Table...")

    # 1. データ読み込み
    try:
        df_top200 = pd.read_csv(TOP200_FILE, dtype={"code": str})
        df_sector = pd.read_csv(SECTOR_FILE, dtype={"code": str})
        df_news_c = pd.read_csv(NEWS_COUNT_FILE, dtype={"Code": str})
        df_news_n = pd.read_csv(NEWS_NAME_FILE, dtype={"Code": str})
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 2. データ結合
    # カラム名統一
    df_top200.columns = ["Code"]
    df_sector = df_sector.rename(columns={"code": "Code"})

    # ベース作成
    merged = df_top200.merge(df_news_n[["Code", "Keyword"]], on="Code", how="left")
    merged = merged.merge(df_sector[["Code", "Sector", "Industry"]], on="Code", how="left")
    merged = merged.merge(df_news_c[["Code", "Articles", "Start_Date", "End_Date"]], on="Code", how="left")

    # 3. 整形
    merged["News Count"] = merged["Articles"].fillna(0).astype(int)
    # 期間フォーマット (例: 2018-2025)
    merged["Data Period"] = merged["Start_Date"].astype(str).str[:4] + "-" + merged["End_Date"].astype(str).str[:4]
    # データタイプ (全銘柄共通で Price + News)
    merged["Data Types"] = "Price, News"

    merged = merged.sort_values("Code").fillna("N/A")

    # 必要なカラムを選択
    final_df = merged[["Code", "Keyword", "Sector", "News Count", "Data Period", "Data Types"]]
    final_df.columns = ["Code", "Company Name", "Sector", "News Articles", "Period", "Data Types"]

    # 4. CSV保存 (確認用)
    csv_path = OUTPUT_DIR / "appendix_stock_list_full.csv"
    final_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # 5. LaTeX (longtable) 生成
    tex_path = OUTPUT_DIR / "stock_list_table.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        # ヘッダー
        f.write(r"\begin{longtable}{clllcl}" + "\n")
        f.write(r"\caption{分析対象銘柄一覧およびデータ概要 (全200銘柄)} \\" + "\n")
        f.write(r"\label{tab:stock_list_full} \\" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Code & Company Name & Sector & News Articles & Period & Data Types \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfirsthead" + "\n")

        # 2ページ目以降のヘッダー
        f.write(r"\multicolumn{6}{c}{{\tablename} \thetable{} -- 続き} \\" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Code & Company Name & Sector & News Articles & Period & Data Types \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n")

        # フッター
        f.write(r"\bottomrule" + "\n")
        f.write(r"\endfoot" + "\n")

        # データ行
        for _, row in final_df.iterrows():
            # 特殊文字のエスケープ (&など)
            name = str(row["Company Name"]).replace("&", r"\&")
            sector = str(row["Sector"]).replace("&", r"\&")

            line = f"{row['Code']} & {name} & {sector} & {row['News Articles']} & {row['Period']} & {row['Data Types']} \\\\"
            f.write(line + "\n")

        f.write(r"\end{longtable}" + "\n")

    print(f"Saved LaTeX: {tex_path}")


if __name__ == "__main__":
    generate_table()
