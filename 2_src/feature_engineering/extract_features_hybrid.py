import asyncio
import json
import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from tqdm.asyncio import tqdm

# ==========================================
# 1. 設定 & 準備
# ==========================================
# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("processing_hybrid.log", encoding="utf-8"), logging.StreamHandler()],
)
logging.getLogger().handlers[1].setLevel(logging.WARNING)

# 環境変数 (Azure OpenAI)
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

# パス設定
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "1_data" / "edinet_reports" / "02_unzipped_files"
OUTPUT_FILE = PROJECT_ROOT / "1_data" / "processed" / "edinet_features_financials_hybrid.csv"

# 同時実行数制限
CONCURRENT_LIMIT = 20

# 分析結果に基づく「正解」設定 (優先順位順)
TARGET_FILE_PATTERNS = ["jpcrp040300", "jpcrp030000"]  # 四半期 -> 有報 の順

TARGET_KEYS_RULE = {
    "NetSales": [
        "jppfs_cor:NetSales",  # 日本基準: 売上高 (7970件)
        "jpcrp_cor:NetSalesSummaryOfBusinessResults",  # サマリー: 売上高 (7827件)
        "jpcrp_cor:RevenueIFRSSummaryOfBusinessResults",  # IFRS: 売上収益 (2472件)
        "jpigp_cor:RevenueIFRS",  # IFRS: 売上収益 (1779件)
        "jppfs_cor:OperatingRevenue1",  # 営業収益 (1255件)
        "jpcrp_cor:RevenuesFromExternalCustomers",  # 外部顧客売上 (5588件) - 保険
        "jppfs_cor:OrdinaryRevenues",  # 経常収益 (銀行等)
    ],
    "OperatingIncome": [
        "jppfs_cor:OperatingIncome",  # 日本基準: 営業利益 (8997件)
        "jpcrp_cor:OperatingIncomeLossSummaryOfBusinessResults",  # サマリー (少ないが念のため)
        "jpigp_cor:OperatingProfitLossIFRS",  # IFRS: 営業利益 (2254件)
        # --- 営業利益がない場合は経常利益 ---
        "jppfs_cor:OrdinaryIncome",  # 日本基準: 経常利益 (9191件) - 最多
        "jpcrp_cor:OrdinaryIncomeLossSummaryOfBusinessResults",  # サマリー: 経常利益 (9179件)
        "jpigp_cor:ProfitLossBeforeTaxIFRS",  # IFRS: 税引前利益 (最終手段)
    ],
}


# ==========================================
# 2. 関数定義: ルールベース抽出 (Python)
# ==========================================
def read_csv_robust(path):
    """エンコーディング対応のCSV読み込み"""
    for enc in ["utf-8", "utf-16", "cp932", "shift_jis"]:
        try:
            return pd.read_csv(path, sep="\t", encoding=enc)
        except:
            try:
                return pd.read_csv(path, sep=",", encoding=enc)
            except:
                continue
    return pd.DataFrame()


def extract_by_rules(folder_path):
    """フォルダ内のCSVからXBRLタグを使って数値を抽出する"""
    extracted = {"NetSales": None, "OperatingIncome": None}

    target_dir = folder_path / "XBRL_TO_CSV"
    if not target_dir.exists():
        all_csvs = list(folder_path.rglob("*.csv"))
    else:
        all_csvs = list(target_dir.glob("*.csv"))

    # ファイル優先順位ソート
    def file_priority(f):
        fname = f.name
        if fname.startswith("jpcrp040300"):
            return 0  # 四半期 (最優先)
        if fname.startswith("jpcrp030000"):
            return 1  # 有報
        return 99

    sorted_csvs = sorted(all_csvs, key=file_priority)

    for csv_file in sorted_csvs:
        # 両方取れたら終了
        if extracted["NetSales"] is not None and extracted["OperatingIncome"] is not None:
            break

        df = read_csv_robust(csv_file)
        if df.empty:
            continue

        cols = df.columns
        id_col = next((c for c in cols if "要素ID" in c or "ElementID" in c), None)
        val_col = next((c for c in cols if "値" in c or "Value" in c), None)
        ctx_col = next((c for c in cols if "コンテキストID" in c or "ContextID" in c), None)

        if not id_col or not val_col:
            continue

        df[id_col] = df[id_col].astype(str)

        for key, tags in TARGET_KEYS_RULE.items():
            if extracted[key] is not None:
                continue

            for tag in tags:
                # 高速化: まずisinで探すが、Prefix違いがあるかもしれないのでcontains
                mask_tag = df[id_col].str.contains(tag, case=False, na=False)
                if not mask_tag.any():
                    continue

                # コンテキストフィルタ (分析結果に基づく)
                if ctx_col:
                    ctx_series = df[ctx_col].astype(str)

                    # 1. CurrentYTDDuration (四半期累計) - 最重要 (7696件)
                    mask_ytd = ctx_series.str.contains("CurrentYTDDuration", case=False, na=False)
                    hits = df[mask_tag & mask_ytd]

                    # 2. CurrentYearDuration (通期) - 次点 (3262件)
                    if hits.empty:
                        mask_year = ctx_series.str.contains("CurrentYearDuration", case=False, na=False)
                        hits = df[mask_tag & mask_year]

                    # 3. Current...Instant (BS項目などは時点)
                    if hits.empty:
                        mask_inst = ctx_series.str.contains("Current.*Instant", case=False, na=False)
                        hits = df[mask_tag & mask_inst]

                    # 4. それでもなければ "Current" を含む何か (最終手段)
                    if hits.empty:
                        mask_curr = ctx_series.str.contains("Current", case=False, na=False)
                        hits = df[mask_tag & mask_curr]
                else:
                    hits = df[mask_tag]

                if not hits.empty:
                    try:
                        val = hits.iloc[0][val_col]
                        val_num = float(str(val).replace(",", ""))
                        extracted[key] = val_num
                        break
                    except:
                        continue
    return extracted


# ==========================================
# 3. 関数定義: LLM抽出 (Azure OpenAI)
# ==========================================
def load_csv_text_for_llm(folder_path):
    """LLM用に重要そうなCSVの先頭だけテキスト化"""
    text_content = ""
    target_dir = folder_path / "XBRL_TO_CSV"
    if not target_dir.exists():
        csv_files = list(folder_path.rglob("*.csv"))
    else:
        csv_files = list(target_dir.glob("*.csv"))

    # 優先ファイルのみ対象
    target_files = [f for f in csv_files if any(p in f.name for p in TARGET_FILE_PATTERNS)]
    if not target_files:
        target_files = csv_files[:3]

    for csv_file in target_files[:2]:  # 最大2ファイルで十分
        try:
            df = read_csv_robust(csv_file)
            if not df.empty:
                cols = df.columns
                # LLMが見やすい列名を探す
                item_col = next((c for c in cols if "項目名" in c or "Label" in c), None)
                val_col = next((c for c in cols if "値" in c or "Value" in c), None)

                if item_col and val_col:
                    use_cols = [item_col, val_col]
                    # 数値が入っている行だけ抽出 (空行やタイトル行を除外)
                    df_valid = df.dropna(subset=[val_col])
                    # 先頭100行
                    text_chunk = df_valid[use_cols].head(100).to_csv(index=False, sep=": ", header=False)
                    text_content += f"\n--- File: {csv_file.name} ---\n{text_chunk}\n"
        except:
            continue
    return text_content[:15000]  # トークン節約


async def extract_by_llm(client, text_data, doc_id):
    """LLMによるフォールバック"""
    system_prompt = """
    あなたは金融アナリストです。以下のXBRLデータ(CSV抜粋)から、企業の業績数値を抽出してください。
    
    【抽出ターゲット】
    1. 売上高 (NetSales, Revenue, 営業収益)
       - 「累計」の数値を優先すること。
    2. 営業利益 (OperatingIncome)
       - ない場合は「経常利益 (OrdinaryIncome)」または「税引前利益」で代用可。
    
    【出力ルール】
    - JSON形式: {"NetSales": 数値, "OperatingIncome": 数値}
    - 単位は「円」(CSVの値はそのまま使うこと)。
    - 不明な場合は null。
    """

    try:
        response = await client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"データ:\n{text_data}"},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        # logging.error(f"LLM Error ({doc_id}): {e}")
        return {"NetSales": None, "OperatingIncome": None}


# ==========================================
# 4. メイン処理
# ==========================================
async def worker(folder_path, client, semaphore):
    async with semaphore:
        doc_id = folder_path.name

        # 1. ルールベース
        try:
            data = await asyncio.to_thread(extract_by_rules, folder_path)
        except Exception:
            data = {"NetSales": None, "OperatingIncome": None}

        method = "Rule"

        # 2. LLMフォールバック (欠損がある場合のみ)
        if (data["NetSales"] is None or data["OperatingIncome"] is None) and client:
            text_data = await asyncio.to_thread(load_csv_text_for_llm, folder_path)
            if text_data.strip():
                llm_data = await extract_by_llm(client, text_data, doc_id)

                if data["NetSales"] is None and llm_data.get("NetSales"):
                    data["NetSales"] = llm_data.get("NetSales")
                    method = "Hybrid"
                if data["OperatingIncome"] is None and llm_data.get("OperatingIncome"):
                    data["OperatingIncome"] = llm_data.get("OperatingIncome")
                    method = "Hybrid"

        return {
            "DocID": doc_id,
            "NetSales": data["NetSales"],
            "OperatingIncome": data["OperatingIncome"],
            "Method": method,
        }


async def main():
    print("--- XBRL財務数値抽出 (Hybrid版 v3: 最適化済) 開始 ---")

    client = None
    if AZURE_OPENAI_API_KEY:
        client = AsyncAzureOpenAI(
            api_key=AZURE_OPENAI_API_KEY, api_version=AZURE_OPENAI_API_VERSION, azure_endpoint=AZURE_OPENAI_ENDPOINT
        )
    else:
        print("※APIキーなし: ルールベースのみで実行")

    doc_folders = [p for p in INPUT_DIR.iterdir() if p.is_dir() and p.name.startswith("S100")]
    print(f"対象フォルダ数: {len(doc_folders)}")

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    tasks = [worker(folder, client, semaphore) for folder in doc_folders]

    results = []
    for f in tqdm.as_completed(tasks, total=len(tasks)):
        res = await f
        if res:
            results.append(res)

    if results:
        df = pd.DataFrame(results)
        df["NetSales"] = pd.to_numeric(df["NetSales"], errors="coerce")
        df["OperatingIncome"] = pd.to_numeric(df["OperatingIncome"], errors="coerce")

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)

        print(f"\n完了: {OUTPUT_FILE}")
        filled = df.dropna(subset=["NetSales", "OperatingIncome"])
        print(f"完全データ数: {len(filled)} / {len(df)} ({len(filled) / len(df):.1%})")
        print(f"LLM救済数: {len(df[df['Method'] == 'Hybrid'])}")


if __name__ == "__main__":
    asyncio.run(main())
