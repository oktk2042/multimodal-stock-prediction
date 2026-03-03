import pandas as pd
from pathlib import Path
import os
import json
import asyncio
import logging
from openai import AsyncAzureOpenAI, RateLimitError
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

# --- ロギング設定 ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.getLogger().handlers[1].setLevel(logging.WARNING)

# --- 環境設定 ---
load_dotenv()
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

# --- ディレクトリ設定 ---
INPUT_DIR = Path("1_data/edinet_reports/02_unzipped_files")
OUTPUT_FILE = Path("1_data/processed/edinet_features_llm.csv")
CONCURRENT_LIMIT = 5 # 同時実行数 (エラーが出る場合は減らしてください)

async def analyze_report(client, text: str, doc_id: str):
    """
    結合されたCSVテキストから財務情報を抽出する
    """
    system_instruction = """
    You are a financial analyst. Analyze the provided CSV content from a Japanese financial report (EDINET).
    Extract the following three items based on the 'Current Fiscal Year' or 'Current Period' (当期/当四半期):
    
    1. NetSales (売上高):
       - Look for "売上高" (Net Sales) or "売上収益" (Revenue).
       - Extract the numeric value for the current period.
    
    2. OperatingIncome (営業利益):
       - Look for "営業利益" (Operating Income) or "営業損失" (Operating Loss).
       - Extract the numeric value. If it is a loss, ensure it is negative (e.g., -100).
    
    3. Sentiment:
       - Analyze the text in sections like "経営者による財政状態、経営成績及びキャッシュ・フローの状況の分析" (Management Discussion and Analysis).
       - Score the management's outlook from -1.0 (Very Pessimistic) to 1.0 (Very Optimistic).
       - 0.0 is Neutral.
    
    Output JSON format only:
    {"NetSales": 12345000000, "OperatingIncome": 500000000, "Sentiment": 0.2}
    
    Rules:
    - Return null if the specific value is clearly not found.
    - Do not include commas in numbers.
    """
    
    # テキストが長すぎる場合は先頭3万文字でカット（gpt-5-miniのコンテキストに合わせて調整）
    input_text = text[:30000] 

    try:
        response = await client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "user", "content": system_instruction + f"\n\n--- CSV DATA START ---\n{input_text}\n--- CSV DATA END ---"}
            ],
            max_completion_tokens=2000,
        )
        
        content = response.choices[0].message.content
        # Markdownタグ削除
        content_cleaned = content.replace("```json", "").replace("```", "").strip()
        
        try:
            data = json.loads(content_cleaned)
            return {
                "DocID": doc_id,
                "NetSales_LLM": data.get("NetSales"),
                "OperatingIncome_LLM": data.get("OperatingIncome"),
                "Sentiment_LLM": data.get("Sentiment")
            }
        except json.JSONDecodeError:
            logging.warning(f"⚠️ {doc_id}: JSON形式エラー")
            return None

    except RateLimitError:
        logging.warning(f"⏳ {doc_id}: レート制限 (Rate Limit)。リトライします...")
        await asyncio.sleep(5) # 少し待つ
        return None
    except Exception as e:
        logging.error(f"❌ {doc_id}: APIエラー - {e}")
        return None

def load_csvs_in_folder(folder_path: Path) -> str:
    """
    指定フォルダ内の XBRL_TO_CSV/*.csv を読み込み（UTF-16優先）
    """
    csv_dir = folder_path / "XBRL_TO_CSV"
    if not csv_dir.exists():
        return ""

    combined_text = []
    # フォルダ内の全CSVを対象
    for csv_file in csv_dir.glob("*.csv"):
        content = None
        # EDINETのCSVは UTF-16 であることが多いので最優先で試す
        encodings_to_try = ["utf-16", "utf-8", "cp932"]
        
        for enc in encodings_to_try:
            try:
                with open(csv_file, "r", encoding=enc) as f:
                    content = f.read()
                break
            except UnicodeDecodeError:
                continue
            except Exception:
                continue

        # 全部だめなら無理やり読む
        if content is None:
            try:
                with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except:
                pass
        
        if content:
            # ファイル名と中身を結合
            combined_text.append(f"--- FILE: {csv_file.name} ---\n{content}\n")
            
    return "\n".join(combined_text)

async def main():
    if not INPUT_DIR.exists():
        logging.error(f"ディレクトリが見つかりません: {INPUT_DIR}")
        return

    # 全フォルダを対象にする
    report_dirs = list(INPUT_DIR.glob("S100*"))
    
    if not report_dirs:
        logging.warning("処理対象のフォルダ（S100...）が見つかりません。")
        return

    logging.info(f"🚀 全件処理を開始します。対象レポート数: {len(report_dirs)} 件")
    
    client = AsyncAzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT
    )

    semaphore = asyncio.Semaphore(CONCURRENT_LIMIT)
    results = []

    async def worker(folder_path):
        async with semaphore:
            try:
                # CSV読み込み
                text = await asyncio.to_thread(load_csvs_in_folder, folder_path)
                
                if not text.strip():
                    return None

                res = await analyze_report(client, text, folder_path.name)
                return res
            except Exception as e:
                return None

    # 全件実行
    tasks = [worker(d) for d in report_dirs]
    
    # 進捗バー付きで実行
    for f in tqdm.as_completed(tasks, total=len(tasks)):
        res = await f
        if res:
            results.append(res)
            # 100件ごとに途中経過をログに出す
            if len(results) % 100 == 0:
                logging.info(f"現在 {len(results)} 件の抽出完了...")

    # 結果保存
    if results:
        df = pd.DataFrame(results)
        # 数値型に変換しておく
        df["NetSales_LLM"] = pd.to_numeric(df["NetSales_LLM"], errors='coerce')
        df["OperatingIncome_LLM"] = pd.to_numeric(df["OperatingIncome_LLM"], errors='coerce')
        
        output_dir = OUTPUT_FILE.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(OUTPUT_FILE, index=False)
        logging.info(f"🎉 全処理完了！ {len(df)} 件のデータを保存しました: {OUTPUT_FILE}")
        print(df.head())
    else:
        logging.warning("有効なデータが1件も抽出できませんでした。")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())