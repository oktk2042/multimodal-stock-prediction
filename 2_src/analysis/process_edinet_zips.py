import re
import zipfile
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# --- 設定 ---
BASE_DIR = Path("C:/M2_Research_Project/1_data/edinet_reports/")
ZIP_DIR = BASE_DIR / "01_zip_files_indices"
META_FILE = BASE_DIR / "00_metadata" / "metadata_2018_2025_all.csv"
OUTPUT_FILE = Path("C:/M2_Research_Project/1_data/processed/extracted_financial_data_indices_parsed.csv")

# 抽出対象のタグ（正規表現で柔軟にマッチさせる）
# ※名前空間プレフィックス(jppfs_cor等)は変動するため、タグの末尾で判定する
TARGET_TAGS = {
    "NetSales": [r"NetSales", r"RevenueIFRS", r"OperatingRevenue1", r"Revenue"],
    "OperatingIncome": [r"OperatingIncome", r"OperatingProfitLossIFRS", r"OperatingProfit"],
    "OrdinaryIncome": [r"OrdinaryIncome"],
    "NetIncome": [r"ProfitLossAttributableToOwnersOfParent", r"CurrentNetIncome"],
    "NetAssets": [r"NetAssets", r"EquityIFRS", r"TotalEquity"],
    "TotalAssets": [r"TotalAssets", r"AssetsIFRS", r"Assets"],
}

# コンテキストの判定（「当期」「連結」などを優先する）
# Prior（前期）やNonConsolidated（個別）を除外するためのキーワード
EXCLUDE_CONTEXTS = ["Prior", "NonConsolidated"]
INCLUDE_CONTEXTS = ["Current", "Consolidated"]  # 必須ではないが、優先度判定に使用


def load_metadata_robust():
    """メタデータ読み込み"""
    print("メタデータを読み込んでいます...")
    if not META_FILE.exists():
        print(f"エラー: {META_FILE} がありません")
        return {}, {}

    try:
        df = pd.read_csv(META_FILE, dtype=str, low_memory=False)
        df = df.dropna(subset=["secCode"])
        df["secCode"] = df["secCode"].str.replace(".0", "", regex=False)
        df["code_4digit"] = df["secCode"].str[:4]

        mapping = dict(zip(df["docID"], df["code_4digit"]))
        date_mapping = dict(zip(df["docID"], df["submitDateTime"]))
        return mapping, date_mapping
    except Exception as e:
        print(f"エラー: {e}")
        return {}, {}


def clean_value(val_str):
    """数値文字列のクリーニング"""
    if not val_str:
        return None
    # カンマ削除
    val_str = val_str.replace(",", "")
    # マイナス記号の統一 (△, ▲ -> -)
    val_str = val_str.replace("△", "-").replace("▲", "-")

    # 数値変換
    try:
        return float(val_str)
    except:
        return None


def parse_ixbrl(html_content):
    """HTML(ixXBRL)コンテンツから財務数値を抽出"""
    soup = BeautifulSoup(html_content, "lxml")
    extracted = {}

    # ix:nonNumeric (テキスト) と ix:nonFraction (数値) を探す
    # 今回は数値データが欲しいので ix:nonFraction を中心に
    tags = soup.find_all(["ix:nonfraction", "ix:nonFraction"])

    for tag in tags:
        name = tag.get("name", "")
        context_ref = tag.get("contextref", "")

        # コンテキストチェック (前期データや個別データを除外)
        # ※簡易的な判定です。厳密にはContext定義を見る必要がありますが、
        # 通常 contextRef="CurrentYearDuration" などの文字列が含まれます。
        if any(ex in context_ref for ex in EXCLUDE_CONTEXTS):
            continue

        # 値の取得 (sign属性でマイナス判定が必要な場合もあるが、まずはテキスト値)
        val_str = tag.text.strip()
        val = clean_value(val_str)

        if val is None:
            continue

        # sign属性の考慮 (sign="-"ならマイナスにする)
        if tag.get("sign", "") == "-":
            val = -abs(val)

        # タグ名マッチング
        for item_key, patterns in TARGET_TAGS.items():
            for pat in patterns:
                # 名前空間(jppfs_cor:)を無視して末尾一致、または含むか判定
                if re.search(f":{pat}$", name) or name == pat:
                    # 既に取得済みの場合、値を上書きするか？
                    # 今回は「桁数が大きい方」を採用する（単位ミスの防止など）
                    # または「コンテキストIDがCurrentYear」に近い方を優先したいが...
                    # 簡易的に「後勝ち」または「最大値」戦略をとる

                    if item_key not in extracted:
                        extracted[item_key] = val
                    else:
                        # 既にある場合、絶対値が大きい方を採用（サマリーと詳細は詳細の方が桁が正確なことが多い）
                        if abs(val) > abs(extracted[item_key]):
                            extracted[item_key] = val

    return extracted


def main():
    # 1. メタデータ
    id_map, date_map = load_metadata_robust()
    if not id_map:
        return

    zip_files = list(ZIP_DIR.glob("*.zip"))
    print(f"処理対象ZIPファイル数: {len(zip_files)}")

    results = []

    # 2. ZIPループ
    for zip_file in tqdm(zip_files, desc="Parsing XBRL"):
        doc_id = zip_file.stem
        code = id_map.get(doc_id)
        if not code:
            continue

        submit_date_str = date_map.get(doc_id, "unknown")
        submit_date = submit_date_str.split()[0] if submit_date_str else "unknown"

        try:
            with zipfile.ZipFile(zip_file, "r") as zf:
                # ixXBRLファイル (.htm) を探す
                # 通常 PublicDoc フォルダ内にある
                target_files = [f for f in zf.namelist() if f.endswith(".htm") and "PublicDoc" in f]

                # 財務データは複数のファイルに分かれていることがあるが、
                # 主要な数値は表紙やサマリー(0000000_header... または 0101010...) にあることが多い
                # 全てパースして結合する

                company_data = {"Code": code, "DocID": doc_id, "SubmissionDate": submit_date}

                extracted_any = False

                for file_in_zip in target_files:
                    with zf.open(file_in_zip) as f:
                        html_content = f.read()

                        # パース実行
                        data = parse_ixbrl(html_content)
                        if data:
                            extracted_any = True
                            # 辞書をマージ（既存より新しい値があれば更新...したいが、
                            # 項目ごとに最大値を採用するロジックにする）
                            for k, v in data.items():
                                if k not in company_data:
                                    company_data[k] = v
                                else:
                                    if abs(v) > abs(company_data[k]):
                                        company_data[k] = v

                if extracted_any:
                    results.append(company_data)

        except Exception as e:
            # print(f"Error {doc_id}: {e}")
            pass

    # 3. 保存
    if results:
        df_result = pd.DataFrame(results)
        print(f"\n抽出データ数: {len(df_result)}")
        print(df_result.head())

        # 必要なカラム順序に整える
        cols = [
            "Code",
            "SubmissionDate",
            "NetSales",
            "OperatingIncome",
            "OrdinaryIncome",
            "NetIncome",
            "NetAssets",
            "TotalAssets",
        ]
        # 存在しないカラムはNaNで埋める
        for c in cols:
            if c not in df_result.columns:
                df_result[c] = None

        df_result = df_result[cols]  # 並べ替え

        df_result.to_csv(OUTPUT_FILE, index=False)
        print(f"保存完了: {OUTPUT_FILE}")
    else:
        print("データが抽出できませんでした。")


if __name__ == "__main__":
    main()
