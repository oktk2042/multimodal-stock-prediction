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
# 1. 実際にモデリングに使用したデータセット (ここからCodeリストを作る)
MODEL_DATA_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
# 2. セクター情報
SECTOR_FILE = DATA_DIR / "stock_sector_info.csv"
# 3. ニュース統計情報 (あれば)
NEWS_STATS_FILE = DATA_DIR / "news_stats_by_code.csv"


def generate_table():
    print("Generating Appendix Table...")

    if not MODEL_DATA_FILE.exists():
        print(f"Error: {MODEL_DATA_FILE} not found.")
        return

    # 1. ターゲット銘柄リストの作成
    # データセットからユニークなCodeとNameを抽出
    df_model = pd.read_csv(
        MODEL_DATA_FILE, dtype={"Code": str, "code": str}, usecols=lambda c: c.lower() in ["code", "name"]
    )
    # カラム統一
    col_map = {c: "code" for c in df_model.columns if c.lower() == "code"}
    col_map.update({c: "Name" for c in df_model.columns if c.lower() == "name"})
    df_model.rename(columns=col_map, inplace=True)

    # 重複排除してリスト化
    df_list = df_model[["code", "Name"]].drop_duplicates().sort_values("code")
    print(f"Target Stocks: {len(df_list)}")

    # 2. セクター情報の結合
    if SECTOR_FILE.exists():
        df_sector = pd.read_csv(SECTOR_FILE, dtype={"Code": str, "code": str})
        if "Code" in df_sector.columns:
            df_sector.rename(columns={"Code": "code"}, inplace=True)

        df_list = pd.merge(df_list, df_sector[["code", "Sector"]], on="code", how="left")
    else:
        df_list["Sector"] = "Unknown"

    # 3. ニュース記事数の結合
    if NEWS_STATS_FILE.exists():
        df_news = pd.read_csv(NEWS_STATS_FILE, dtype={"Code": str, "code": str})
        if "Code" in df_news.columns:
            df_news.rename(columns={"Code": "code"}, inplace=True)

        # 必要なカラム: Articles (記事数)
        if "Articles" in df_news.columns:
            df_list = pd.merge(df_list, df_news[["code", "Articles"]], on="code", how="left")
            df_list["Articles"] = df_list["Articles"].fillna(0).astype(int)
        else:
            df_list["Articles"] = "-"
    else:
        df_list["Articles"] = "-"

    # 4. LaTeXテーブル作成
    tex_path = OUTPUT_DIR / "appendix_stock_list.tex"

    with open(tex_path, "w", encoding="utf-8") as f:
        # Preamble
        f.write(r"\begin{longtable}{cllc}" + "\n")
        f.write(r"\caption{分析対象銘柄一覧 (Top 200) およびニュース記事数} \\" + "\n")
        f.write(r"\label{tab:stock_list_appendix} \\" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Code & Company Name & Sector & News Articles \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endfirsthead" + "\n")

        # Header for next pages
        f.write(r"\multicolumn{4}{c}{{\tablename} \thetable{} -- 続き} \\" + "\n")
        f.write(r"\toprule" + "\n")
        f.write(r"Code & Company Name & Sector & News Articles \\" + "\n")
        f.write(r"\midrule" + "\n")
        f.write(r"\endhead" + "\n")

        # Footer
        f.write(r"\bottomrule" + "\n")
        f.write(r"\endfoot" + "\n")

        # Data Rows
        for _, row in df_list.iterrows():
            code = row["code"]
            name = row["Name"] if pd.notna(row["Name"]) else "Unknown"
            # LaTeXエスケープ処理 (＆など)
            name = name.replace("&", r"\&")
            sector = row["Sector"] if pd.notna(row["Sector"]) else "-"
            articles = row["Articles"]

            f.write(f"{code} & {name} & {sector} & {articles} \\\\" + "\n")

        f.write(r"\end{longtable}" + "\n")

    print(f"LaTeX table saved to: {tex_path}")

    # CSVとしても保存（確認用）
    csv_path = OUTPUT_DIR / "appendix_stock_list_check.csv"
    df_list.to_csv(csv_path, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    generate_table()
