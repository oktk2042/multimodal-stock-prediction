from pathlib import Path

import pandas as pd

# ==========================================
# 設定
# ==========================================
# プロジェクトルート (環境に合わせて調整してください)
PROJECT_ROOT = Path(".").resolve()
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 入力ファイル
PRICE_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
# アップロードいただいたファイルを指定
NEWS_FILE = DATA_DIR / "sentiment_noise_check_list.csv"


def find_missing_cases():
    print("データ読み込み中...")

    if not PRICE_FILE.exists():
        print(f"Error: {PRICE_FILE} not found.")
        return

    # 1. 株価データの読み込み
    # DtypeWarning回避のため文字列として読み込み、後で型変換
    df_price = pd.read_csv(PRICE_FILE, dtype={"Code": str, "code": str}, low_memory=False)

    # カラム名統一 (Code -> code)
    if "Code" in df_price.columns:
        df_price.rename(columns={"Code": "code"}, inplace=True)

    # 日付変換
    df_price["Date"] = pd.to_datetime(df_price["Date"])

    # 2. 5日後リターンの計算 (もしカラムがなければ)
    if "Target_Return_5D" not in df_price.columns:
        print("Calculating 5-day returns...")
        df_price.sort_values(["code", "Date"], inplace=True)
        df_price["Close"] = pd.to_numeric(df_price["Close"], errors="coerce")
        df_price["Target_Return_5D"] = df_price.groupby("code")["Close"].shift(-5) / df_price["Close"] - 1

    # 3. ニュースデータの読み込み
    if NEWS_FILE.exists():
        print(f"ニュース辞書を読み込んでいます: {NEWS_FILE.name}")
        df_news = pd.read_csv(NEWS_FILE, dtype={"Code": str, "code": str}, low_memory=False)

        # カラム名統一 (Code -> code)
        if "Code" in df_news.columns:
            df_news.rename(columns={"Code": "code"}, inplace=True)

        df_news["Date"] = pd.to_datetime(df_news["Date"])

        # 結合用に重複排除 (1日1銘柄で最もスコアの絶対値が大きいものを採用)
        if "News_Sentiment" in df_news.columns:
            df_news["News_Sentiment"] = pd.to_numeric(df_news["News_Sentiment"], errors="coerce")
            df_news["abs_score"] = df_news["News_Sentiment"].abs()
            df_news_best = df_news.sort_values("abs_score", ascending=False).drop_duplicates(subset=["Date", "code"])

            # マージ (左結合)
            df_merged = pd.merge(df_price, df_news_best[["Date", "code", "Title"]], on=["Date", "code"], how="left")
        else:
            print("News_Sentiment column not found in news file.")
            df_merged = df_price.copy()
            df_merged["Title"] = ""
    else:
        print("ニュース詳細ファイルが見つかりません。タイトルなしで進めます。")
        df_merged = df_price.copy()
        df_merged["Title"] = ""

    # スコア列の特定 (Gate_Score優先)
    if "Gate_Score" in df_merged.columns:
        score_col = "Gate_Score"
    elif "News_Sentiment" in df_merged.columns:
        score_col = "News_Sentiment"
    else:
        # どちらもなければFinBERTスコアを使う
        score_col = "FinBERT_Score" if "FinBERT_Score" in df_merged.columns else None

    if not score_col:
        print("Error: No score column found (Gate_Score, News_Sentiment, FinBERT_Score).")
        return

    print(f"Using score column: {score_col}")

    # ----------------------------------------------------------
    # 1. Negative Cases (閾値を -0.2 に緩和)
    # ----------------------------------------------------------
    # 条件: スコアがネガティブ(-0.2以下) かつ リターンがマイナス(-5%以下)
    neg_cases = df_merged[(df_merged[score_col] < -0.2) & (df_merged["Target_Return_5D"] < -0.05)].sort_values(
        "Target_Return_5D", ascending=True
    )

    print("\n=== Negative Cases (Relaxed Threshold) ===")
    if not neg_cases.empty:
        cols = ["Date", "code", "Name", score_col, "Target_Return_5D", "Title"]
        disp_cols = [c for c in cols if c in neg_cases.columns]
        print(neg_cases[disp_cols].head(5).to_string(index=False))

        save_path = OUTPUT_DIR / "case_study_negative_relaxed.csv"
        neg_cases.head(20).to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f"Saved to {save_path}")
    else:
        print("Not found.")

    # ----------------------------------------------------------
    # 2. Gate Closed Cases (ノイズ遮断)
    # ----------------------------------------------------------
    # 条件: 株価変動は大 (|Return| > 5%) だが、Gateは閉じている (|Score| < 0.25)
    # Gate Scoreがある場合のみ実行
    if score_col == "Gate_Score":
        closed_cases = df_merged[
            (df_merged[score_col].abs() < 0.25) & (df_merged["Target_Return_5D"].abs() > 0.05)
        ].sort_values("Target_Return_5D", key=abs, ascending=False)

        print("\n=== Gate Closed Cases (Noise Filtering) ===")
        if not closed_cases.empty:
            cols = ["Date", "code", "Name", score_col, "Target_Return_5D", "Title"]
            disp_cols = [c for c in cols if c in closed_cases.columns]
            print(closed_cases[disp_cols].head(5).to_string(index=False))

            save_path = OUTPUT_DIR / "case_study_gate_closed.csv"
            closed_cases.head(20).to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"Saved to {save_path}")
        else:
            print("Not found.")
    else:
        print("\nGate Score not available, skipping Gate Closed analysis.")


if __name__ == "__main__":
    find_missing_cases()
