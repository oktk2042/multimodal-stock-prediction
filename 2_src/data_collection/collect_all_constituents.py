import os
import warnings
from io import StringIO

import pandas as pd
import requests
import vertexai

warnings.simplefilter("ignore")

# --- 設定 ---
SAVE_DIR = "1_data/raw/"
os.makedirs(SAVE_DIR, exist_ok=True)
PROJECT_ID = "m2-stock-research"
vertexai.init(project=PROJECT_ID)

# from dotenv import load_dotenv
# import google.generativeai as genai
# .envファイルの読み込み
# load_dotenv()
# Google AI Studioで取得したご自身のAPIキーを設定してください
# GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')
# genai.configure(api_key=GOOGLE_API_KEY)

# --- ソースファイルの定義 ---
TOPIX_CSV_PATH = "topixweight_j.csv"
JPX400_CSV_PATH = "400_j.csv"
GROWTH_CORE_CSV_PATH = "gcore_top20.csv"
GROWTH_250_CSV_PATH = "growth257.csv"
NIKKEI225_URL = "https://indexes.nikkei.co.jp/nkave/index/component?idx=nk225"


def save_tickers_to_file(tickers, filename):
    save_path = os.path.join(SAVE_DIR, filename)
    tickers_with_suffix = [f"{str(ticker)}.T" for ticker in tickers]
    unique_tickers = sorted(list(set(tickers_with_suffix)))

    with open(save_path, "w") as f:
        f.write("\n".join(unique_tickers))
    print(f" -> {len(unique_tickers)} 銘柄を {save_path} に保存しました。")


def extract_from_nikkei_url():
    print("--- Nikkei 225 をURLから取得中 ---")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = requests.get(NIKKEI225_URL, headers=headers)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        rename_map = {"コード": "code", "銘柄名": "name", "社名": "company"}

        all_records = []
        for table in tables:
            if "コード" in table.columns or "コード " in table.columns:
                df = table.rename(columns={"コード ": "コード"})
                if "銘柄名" not in df.columns:
                    continue
                if "社名" not in df.columns:
                    df["社名"] = None

                tmp = df[["コード", "銘柄名", "社名"]].dropna(subset=["コード", "銘柄名"])
                tmp = tmp.rename(columns=rename_map)
                tmp["code"] = tmp["code"].astype(int).astype(str).str.zfill(4)
                tmp["index"] = "Nikkei 225"
                cols = ["index"] + [col for col in tmp.columns if col != "index"]
                tmp = tmp[cols]
                all_records.append(tmp)

        if not all_records:
            print("警告: Nikkei 225 の表から銘柄を抽出できませんでした。")
            return

        records = pd.concat(all_records, ignore_index=True)
        save_path = os.path.join(SAVE_DIR, "nikkei_225.csv")
        records.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f" -> {len(records)} 件を {save_path} に保存しました。")

    except Exception as e:
        print(f"エラー: Nikkei 225 の取得中に問題が発生しました - {e}")


def extract_from_topix_csv():
    print("--- TOPIX Core30 & 100 をCSVから取得中 ---")
    try:
        df = pd.read_csv(TOPIX_CSV_PATH, encoding="utf-8", header=0)

        # 共通で使うカラムを揃える
        base_cols = ["コード", "銘柄名", "業種", "TOPIXに占める個別銘柄のウエイト"]
        rename_map = {
            "コード": "code",
            "銘柄名": "name",
            "業種": "industry",
            "TOPIXに占める個別銘柄のウエイト": "weight",
        }

        # Core30
        core30_df = df[df["ニューインデックス区分"] == "TOPIX Core30"]
        records_core30 = core30_df[base_cols].rename(columns=rename_map)
        records_core30["index"] = "TOPIX Core30"
        cols_core30 = ["index"] + [col for col in records_core30.columns if col != "index"]
        records_core30 = records_core30[cols_core30]
        save_path = os.path.join(SAVE_DIR, "topix_core30.csv")
        records_core30.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f" -> {len(records_core30)} 件を {save_path} に保存しました。")

        # TOPIX100（Core30 + Large70）
        topix100_df = df[df["ニューインデックス区分"].isin(["TOPIX Core30", "TOPIX Large70"])]
        records_topix100 = topix100_df[base_cols].rename(columns=rename_map)
        records_topix100["index"] = "TOPIX 100"
        cols_topix100 = ["index"] + [col for col in records_topix100.columns if col != "index"]
        records_topix100 = records_topix100[cols_topix100]
        save_path = os.path.join(SAVE_DIR, "topix_100.csv")
        records_topix100.to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f" -> {len(records_topix100)} 件を {save_path} に保存しました。")

    except Exception as e:
        print(f"エラー: TOPIX CSVの処理中に問題が発生しました - {e}")


def extract_from_jpx400_csv(csv_path, output_filename, index_name):
    print(f"--- {index_name} をCSVから取得中 ---")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", header=0)
        records = []
        for _, row in df.iterrows():
            try:
                code = str(row["コード"]).strip().zfill(4)
                market = str(row["市場区分"]).strip() if "市場区分" in row else None
                name = str(row["銘柄名"]).strip()
                count = int(row["累積採用回数"]) if "累積採用回数" in row and pd.notna(row["累積採用回数"]) else None
                records.append({"index": index_name, "code": code, "name": name, "market": market, "count": count})
            except ValueError:
                continue

        if not records:
            print(f"警告: {index_name} のCSVから銘柄データを抽出できませんでした。")
            return

        save_path = os.path.join(SAVE_DIR, output_filename)
        pd.DataFrame(records).to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f" -> {len(records)} 件を {save_path} に保存しました。")

    except Exception as e:
        print(f"エラー: {index_name} CSVの処理中に問題が発生しました - {e}")


def extract_from_growth_csv(csv_path, output_filename, index_name):
    print(f"--- {index_name} をCSVから取得中 ---")
    try:
        df = pd.read_csv(csv_path, encoding="utf-8", header=0)

        records = []
        for _, row in df.iterrows():
            code_str = str(row["コード"]).strip()
            name = str(row["銘柄名"]).strip()
            if "A" in code_str:
                code = code_str
            else:
                code = code_str.zfill(4)
            records.append({"index": index_name, "code": code, "name": name})

        if not records:
            print(f"警告: {index_name} のCSVから銘柄データを抽出できませんでした。")
            return

        save_path = os.path.join(SAVE_DIR, output_filename)
        pd.DataFrame(records).to_csv(save_path, index=False, encoding="utf-8-sig")
        print(f" -> {len(records)} 件を {save_path} に保存しました。")

    except Exception as e:
        print(f"エラー: {index_name} CSVの処理中に問題が発生しました - {e}")


if __name__ == "__main__":
    extract_from_nikkei_url()
    extract_from_topix_csv()
    extract_from_jpx400_csv(JPX400_CSV_PATH, "jpx_nikkei_400.csv", "JPX-Nikkei 400")  # 関数とパスを変更
    extract_from_growth_csv(GROWTH_CORE_CSV_PATH, "growth_core.csv", "TSE Growth Core")
    extract_from_growth_csv(GROWTH_250_CSV_PATH, "growth_250.csv", "TSE Growth 250")

    print("\nすべての処理が完了しました。")
