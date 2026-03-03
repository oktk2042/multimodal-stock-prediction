import zipfile
import random
from pathlib import Path
import os

# --- 設定 ---
# ZIPファイルが保存されているフォルダ
ZIP_DIR = Path("C:/M2_Research_Project/1_data/edinet_reports/01_zip_files_indices")

def analyze_zip_content():
    if not ZIP_DIR.exists():
        print(f"フォルダが見つかりません: {ZIP_DIR}")
        return

    # ZIPファイルをリストアップ
    all_zips = list(ZIP_DIR.glob("*.zip"))
    total_zips = len(all_zips)
    print(f"フォルダ内のZIPファイル総数: {total_zips}")

    if total_zips == 0:
        return

    # ランダムに5つ選んで中身を見る（ファイル数が少なければ全数）
    sample_size = min(5, total_zips)
    samples = random.sample(all_zips, sample_size)

    print("\n" + "="*50)
    print("ZIPファイル構造調査レポート")
    print("="*50)

    for zip_path in samples:
        print(f"\n📁 ファイル名: {zip_path.name}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 中身のファイルリストを取得
                file_list = zf.namelist()
                
                # 統計
                csv_count = sum(1 for f in file_list if f.lower().endswith('.csv'))
                xbrl_count = sum(1 for f in file_list if f.lower().endswith('.xbrl'))
                
                print(f"   - 総ファイル数: {len(file_list)}")
                print(f"   - CSVファイル数: {csv_count}")
                print(f"   - XBRLファイル数: {xbrl_count}")
                
                # 階層構造のサンプル表示 (先頭100件)
                print("   - 含まれるファイル例 (Top 100):")
                for f in file_list[:100]:
                    print(f"     Running... {f}")
                    
                # もしCSVがあれば、そのパスを表示（どこにあるか？）
                if csv_count > 0:
                    print("   - 【発見】CSVファイルの場所例:")
                    csv_files = [f for f in file_list if f.lower().endswith('.csv')]
                    for f in csv_files[:3]:
                        print(f"     -> {f}")
                else:
                    print("   - ⚠️ CSVファイルは含まれていません。")

        except zipfile.BadZipFile:
            print("   - ❌ 壊れたZIPファイルです")
        except Exception as e:
            print(f"   - ❌ エラー: {e}")

if __name__ == "__main__":
    analyze_zip_content()