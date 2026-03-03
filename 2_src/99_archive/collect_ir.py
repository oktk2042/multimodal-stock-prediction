import os
import requests
from bs4 import BeautifulSoup
import urllib.parse

# --- 設定 ---
# ターゲットURL（例：トヨタのIRニュースページ）
BASE_URL = "https://global.toyota/jp/ir/news/"
SAVE_DIR = "../../1_data/raw/ir_reports/toyota"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- 収集処理 ---
def scrape_toyota_ir():
    print("トヨタのIR資料の収集を開始します...")
    try:
        response = requests.get(BASE_URL)
        response.raise_for_status() # エラーチェック

        soup = BeautifulSoup(response.content, 'html.parser')

        # サイトのHTML構造に合わせてCSSセレクタを記述する
        # (例: 'a'タグで、クラス名が 'c-link-text--pdf' のものを探す)
        pdf_links = soup.select('a.c-link-text--pdf') 

        print(f"{len(pdf_links)} 件のPDFリンクが見つかりました。")

        for link in pdf_links:
            pdf_title = link.text.strip()
            pdf_relative_url = link.get('href')
            
            # 絶対URLに変換
            pdf_url = urllib.parse.urljoin(BASE_URL, pdf_relative_url)
            
            # PDFをダウンロード
            pdf_response = requests.get(pdf_url)
            
            # ファイル名をサニタイズ（ファイル名に使えない文字を削除）
            safe_filename = "".join(c for c in pdf_title if c.isalnum() or c in (' ', '.', '_')).rstrip()
            save_path = os.path.join(SAVE_DIR, f"{safe_filename}.pdf")
            
            with open(save_path, 'wb') as f:
                f.write(pdf_response.content)
            print(f" -> {save_path} にダウンロードしました。")

    except Exception as e:
        print(f"エラー: スクレイピング中に問題が発生しました - {e}")

if __name__ == "__main__":
    scrape_toyota_ir()