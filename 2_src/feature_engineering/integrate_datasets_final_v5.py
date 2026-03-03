from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ==========================================
# 1. パス設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
META_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "00_metadata"

# --- 入力ファイル群 ---
# 1. ベースデータ (株価・テクニカル)
BASE_PRICE_FILE = DATA_DIR / "stock_data_features_v1.csv"

# 2. 財務・ファンダメンタルズ
FINANCIALS_FILE = DATA_DIR / "extracted_financial_data_indices_parsed.csv"  # 数値
FINBERT_FILE = DATA_DIR / "edinet_features_finbert_indices_strict.csv"  # 感情スコア
METADATA_FILE = META_DIR / "metadata_2018_2025_all.csv"  # 紐付け用

# 3. 外部データ
NEWS_FILE = DATA_DIR / "news_sentiment_historical.csv"
SECTOR_FILE = DATA_DIR / "stock_sector_info.csv"
MACRO_FILE = DATA_DIR / "global_macro_features.csv"

# --- 出力設定 ---
OUTPUT_DIR_YEARLY = DATA_DIR / "final_datasets_yearly"
OUTPUT_FILE_FULL = DATA_DIR / "final_modeling_dataset_v5.csv"


# ==========================================
# 2. ユーティリティ
# ==========================================
def read_csv_safe(file_path, dtype=None):
    if not file_path.exists():
        print(f"⚠️ ファイルが見つかりません: {file_path}")
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path, dtype=dtype, low_memory=False)
    except Exception as e:
        print(f"❌ 読み込みエラー {file_path.name}: {e}")
        return pd.DataFrame()


def main():
    print("--- 最終データセット統合開始 ---")

    # ---------------------------------------------------------
    # Step 1: FinBERTスコアに「Code/Date」を付与
    # ---------------------------------------------------------
    print("\n1. FinBERTスコアの紐付け処理...")
    df_score = read_csv_safe(FINBERT_FILE, dtype={"DocID": str})
    df_meta = read_csv_safe(METADATA_FILE, dtype={"docID": str, "secCode": str})

    if df_score.empty or df_meta.empty:
        print("❌ スコアまたはメタデータが不足しています。")
        return

    # メタデータ整形
    df_meta.rename(columns={"docID": "DocID", "submitDateTime": "SubmissionDate"}, inplace=True)
    df_meta = df_meta.dropna(subset=["secCode"])
    df_meta["Code"] = df_meta["secCode"].str.replace(".0", "", regex=False).str[:4]
    df_meta["SubmissionDate"] = pd.to_datetime(df_meta["SubmissionDate"]).dt.normalize()

    # 結合
    df_meta_map = df_meta[["DocID", "Code", "SubmissionDate"]].drop_duplicates()
    df_score_mapped = pd.merge(df_score, df_meta_map, on="DocID", how="inner")

    # 重複排除
    df_score_agg = df_score_mapped.groupby(["Code", "SubmissionDate"])["FinBERT_Score"].mean().reset_index()
    print(f"   -> 紐付け完了: {len(df_score_agg):,} 件")

    # ---------------------------------------------------------
    # Step 2: 財務数値とセンチメントの統合
    # ---------------------------------------------------------
    print("\n2. 財務数値とセンチメントを統合中...")
    df_fin = read_csv_safe(FINANCIALS_FILE, dtype={"Code": str})

    if not df_fin.empty:
        df_fin["SubmissionDate"] = pd.to_datetime(df_fin["SubmissionDate"])
        # CodeとDateで結合
        df_fundamentals = pd.merge(df_fin, df_score_agg, on=["Code", "SubmissionDate"], how="outer")
    else:
        df_fundamentals = df_score_agg

    # カラム名整理 (Fin_接頭辞)
    exclude_cols = ["Code", "SubmissionDate", "DocID"]
    rename_map = {c: f"Fin_{c}" for c in df_fundamentals.columns if c not in exclude_cols}
    df_fundamentals.rename(columns=rename_map, inplace=True)

    # 「日付順」にソート
    df_fundamentals.sort_values("SubmissionDate", inplace=True)
    print(f"   -> ファンダメンタルズ総数: {len(df_fundamentals):,} 行")

    # ---------------------------------------------------------
    # Step 3: 株価データへの統合 (Merge Asof)
    # ---------------------------------------------------------
    print("\n3. 株価データへ統合中...")
    df_price = read_csv_safe(BASE_PRICE_FILE, dtype={"code": str, "Code": str})
    if "code" in df_price.columns:
        df_price.rename(columns={"code": "Code"}, inplace=True)

    df_price["Date"] = pd.to_datetime(df_price["Date"])
    df_price.sort_values("Date", inplace=True)

    # マージ (backward)
    df_merged = pd.merge_asof(
        df_price, df_fundamentals, left_on="Date", right_on="SubmissionDate", by="Code", direction="backward"
    )

    # 欠損埋め
    fin_cols = [c for c in df_merged.columns if c.startswith("Fin_")]
    df_merged[fin_cols] = df_merged[fin_cols].fillna(0)

    # ---------------------------------------------------------
    # Step 4: ニュース・マクロ・セクターの統合
    # ---------------------------------------------------------
    print("\n4. ニュース・マクロ・セクター情報を統合中...")

    # News
    df_news = read_csv_safe(NEWS_FILE, dtype={"code": str, "Code": str})
    if not df_news.empty:
        if "code" in df_news.columns:
            df_news.rename(columns={"code": "Code"}, inplace=True)
        df_news["Date"] = pd.to_datetime(df_news["Date"])

        # 個別ニュース
        df_indiv = (
            df_news[df_news["Code"] != "9999"]
            .groupby(["Code", "Date"])
            .agg({"News_Sentiment": "mean", "Title": "count"})
            .reset_index()
            .rename(columns={"Title": "News_Count"})
        )

        # 市況ニュース
        df_market = (
            df_news[df_news["Code"] == "9999"]
            .groupby("Date")["News_Sentiment"]
            .mean()
            .reset_index()
            .rename(columns={"News_Sentiment": "Market_News_Sentiment"})
        )

        df_merged = pd.merge(df_merged, df_indiv, on=["Code", "Date"], how="left")
        df_merged = pd.merge(df_merged, df_market, on="Date", how="left")

        # 欠損埋め
        for c in ["News_Sentiment", "News_Count", "Market_News_Sentiment"]:
            if c in df_merged.columns:
                df_merged[c] = df_merged[c].fillna(0)

    # Macro (1日ラグ)
    df_macro = read_csv_safe(MACRO_FILE)
    if not df_macro.empty:
        df_macro["Date"] = pd.to_datetime(df_macro["Date"]) + pd.Timedelta(days=1)
        df_merged = pd.merge(df_merged, df_macro, on="Date", how="left")

        # 直前の値で埋める
        macro_cols = [c for c in df_macro.columns if c != "Date"]
        df_merged[macro_cols] = df_merged[macro_cols].ffill()

    # Sector & Valuation
    df_sector = read_csv_safe(SECTOR_FILE, dtype={"code": str, "Code": str})
    if not df_sector.empty:
        if "code" in df_sector.columns:
            df_sector.rename(columns={"code": "Code"}, inplace=True)
        cols = ["Code", "Sector", "MarketCap"]
        df_merged = pd.merge(df_merged, df_sector[[c for c in cols if c in df_sector.columns]], on="Code", how="left")

        # 指標計算
        mcap_col = "MarketCap"
        if mcap_col in df_merged.columns:
            if "Fin_NetAssets" in df_merged.columns:
                df_merged["Val_PBR"] = df_merged[mcap_col] / df_merged["Fin_NetAssets"].replace(0, np.nan)
            if "Fin_NetIncome" in df_merged.columns:
                df_merged["Val_PER"] = df_merged[mcap_col] / df_merged["Fin_NetIncome"].replace(0, np.nan)
            if "Fin_NetSales" in df_merged.columns:
                df_merged["Val_PSR"] = df_merged[mcap_col] / df_merged["Fin_NetSales"].replace(0, np.nan)

        # One-Hot Encoding
        if "Sector" in df_merged.columns:
            df_merged = pd.get_dummies(df_merged, columns=["Sector"], prefix="Sec", dummy_na=True)

    # ---------------------------------------------------------
    # Step 5: 保存
    # ---------------------------------------------------------
    print("\n5. 保存処理...")
    OUTPUT_DIR_YEARLY.mkdir(parents=True, exist_ok=True)

    # 掃除
    drop_cols = ["SubmissionDate", "DocID", "Fin_DocID"]
    df_merged.drop(columns=[c for c in drop_cols if c in df_merged.columns], inplace=True)

    # 全期間保存
    df_merged.to_csv(OUTPUT_FILE_FULL, index=False, encoding="utf-8-sig")
    print(f"   -> 全データ保存完了: {OUTPUT_FILE_FULL}")

    # 年別分割保存
    df_merged["Year"] = df_merged["Date"].dt.year
    for year in tqdm(sorted(df_merged["Year"].dropna().unique().astype(int)), desc="Yearly Split"):
        df_year = df_merged[df_merged["Year"] == year].drop(columns=["Year"])
        df_year.to_csv(OUTPUT_DIR_YEARLY / f"final_data_{year}.csv", index=False, encoding="utf-8-sig")

    print("\n✅ 全工程完了。分析用データセットが完成しました。")
    print(f"   最終行数: {len(df_merged)}")


if __name__ == "__main__":
    main()
