import re
from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル
INPUT_FILE = DATA_DIR / "extracted_financial_data.csv"
# 出力ファイル
OUTPUT_FILE = DATA_DIR / "financial_data_wide.csv"

# ==========================================
# マッピング定義 (ElementID -> 標準項目名)
# ==========================================
# 優先度の高いタグから順にマッピング
TAG_MAPPING = {
    # 売上高
    "jppfs_cor:NetSales": "NetSales",
    "jpigp_cor:RevenueIFRS": "NetSales",
    "jpcrp_cor:NetSalesSummaryOfBusinessResults": "NetSales",
    "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults": "NetSales",
    # 営業利益
    "jppfs_cor:OperatingIncome": "OperatingIncome",
    "jpigp_cor:OperatingProfitLossIFRS": "OperatingIncome",
    "jpcrp_cor:OperatingIncomeSummaryOfBusinessResults": "OperatingIncome",
    "jpcrp_cor:OperatingProfitLossIFRSSummaryOfBusinessResults": "OperatingIncome",
    # 経常利益 (IFRSには概念がないが、税引前利益で代用するか要検討。ここでは分離)
    "jppfs_cor:OrdinaryIncome": "OrdinaryIncome",
    "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults": "OrdinaryIncome",
    # 税引前利益 (IFRS)
    "jpigp_cor:ProfitLossBeforeTaxIFRS": "ProfitBeforeTax",
    "jpcrp_cor:ProfitLossBeforeTaxIFRSSummaryOfBusinessResults": "ProfitBeforeTax",
    # 当期純利益
    "jppfs_cor:ProfitLossAttributableToOwnersOfParent": "NetIncome",
    "jpigp_cor:ProfitLossAttributableToOwnersOfParentIFRS": "NetIncome",
    "jpcrp_cor:ProfitLossAttributableToOwnersOfParentSummaryOfBusinessResults": "NetIncome",
    "jpcrp_cor:ProfitLossAttributableToOwnersOfParentIFRSSummaryOfBusinessResults": "NetIncome",
    # 総資産
    "jppfs_cor:TotalAssets": "TotalAssets",
    "jpigp_cor:AssetsIFRS": "TotalAssets",
    "jpcrp_cor:TotalAssetsSummaryOfBusinessResults": "TotalAssets",
    "jpcrp_cor:TotalAssetsIFRSSummaryOfBusinessResults": "TotalAssets",
    # 純資産
    "jppfs_cor:NetAssets": "NetAssets",
    "jpigp_cor:EquityIFRS": "NetAssets",
    "jpcrp_cor:NetAssetsSummaryOfBusinessResults": "NetAssets",
    "jpcrp_cor:EquityIFRSSummaryOfBusinessResults": "NetAssets",
    # 現金
    "jppfs_cor:CashAndDeposits": "Cash",
    "jpigp_cor:CashAndCashEquivalentsIFRS": "Cash",
}

# キーワードベースの予備マッピング（ElementIDが一致しない場合用）
KEYWORD_MAPPING = [
    ("売上", "NetSales"),
    ("収益", "NetSales"),  # IFRS
    ("営業利益", "OperatingIncome"),
    ("経常利益", "OrdinaryIncome"),
    ("当期純利益", "NetIncome"),
    ("当期利益", "NetIncome"),
    ("総資産", "TotalAssets"),
    ("純資産", "NetAssets"),
    ("現金", "Cash"),
]


def extract_date_from_filename(filename):
    """ファイル名から決算期末日を抽出する"""
    # パターン: DocID_Type_Code_YYYY-MM-DD_...
    # 例: ..._2023-12-31_01_2024-03-27.csv -> 2023-12-31
    match = re.search(r"_(\d{4}-\d{2}-\d{2})_", filename)
    if match:
        return match.group(1)
    return None


def standardize_item_name(row):
    """ElementIDまたはItemNameから標準項目名を決定する"""
    eid = str(row["ElementID"])

    # 1. ElementIDでマッピング
    if eid in TAG_MAPPING:
        return TAG_MAPPING[eid]

    # 2. マッピングに含まれるが接尾辞が違う場合などを考慮 (部分一致)
    # 例: ...SummaryOfBusinessResults など
    for tag, std_name in TAG_MAPPING.items():
        if tag in eid:
            return std_name

    # 3. ItemNameでキーワード検索
    name = str(row["ItemName"])
    for kw, std_name in KEYWORD_MAPPING:
        if kw in name:
            return std_name

    return None


def main():
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at {INPUT_FILE}")
        return

    print("Loading extracted financial data...")
    # low_memory=False はデータ型推論の警告抑制
    df = pd.read_csv(INPUT_FILE, dtype={"Code": str}, low_memory=False)

    print(f"Original records: {len(df)}")

    # 1. 数値のクリーニング
    # 全角数字やカンマの除去
    def clean_number(x):
        if pd.isna(x):
            return np.nan
        s = str(x).replace(",", "").strip()
        # マイナス記号の統一
        s = s.replace("△", "-").replace("▲", "-")
        try:
            return float(s)
        except:
            return np.nan

    df["ValueClean"] = df["Value"].apply(clean_number)
    df = df.dropna(subset=["ValueClean"])

    # 2. ContextIDによるフィルタリング
    # "Prior" (前期比較用) が含まれるものを除外
    # ただし、ファイル名から「その期の」データを取ろうとしているので、
    # 同じファイル内の「前期」データは不要（その期の過去ファイルがあるはずだから）
    if "ContextID" in df.columns:
        mask_current = ~df["ContextID"].astype(str).str.contains("Prior", case=False, na=False)
        df = df[mask_current]

    # 3. 項目名の標準化
    df["StandardItem"] = df.apply(standardize_item_name, axis=1)
    df = df.dropna(subset=["StandardItem"])

    # 4. 決算期（PeriodEnd）の抽出
    df["PeriodEnd"] = df["File"].apply(extract_date_from_filename)
    df = df.dropna(subset=["PeriodEnd"])

    print(f"Records after cleaning: {len(df)}")

    # 5. ピボットテーブルの作成
    # Code, PeriodEnd をキーにして、StandardItem をカラムにする
    # 重複がある場合（同じ項目が複数タグで取れている場合）、最大値を取るなどの集約が必要
    # ※通常、経営指標と財務諸表で同じ値が入るため、maxやfirstで問題ない

    df_wide = df.pivot_table(
        index=["Code", "PeriodEnd"], columns="StandardItem", values="ValueClean", aggfunc="max"
    ).reset_index()

    # 日付でソート
    df_wide["PeriodEnd"] = pd.to_datetime(df_wide["PeriodEnd"])
    df_wide = df_wide.sort_values(["Code", "PeriodEnd"])

    print("\n--- Reshaped Data Sample ---")
    print(df_wide.head())
    print(f"\nColumns: {df_wide.columns.tolist()}")
    print(f"Total Rows (Company-Periods): {len(df_wide)}")

    # 保存
    df_wide.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved extracted wide data to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
