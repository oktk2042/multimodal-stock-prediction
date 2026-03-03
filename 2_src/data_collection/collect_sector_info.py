import time
from pathlib import Path

import pandas as pd
import yfinance as yf
from tqdm import tqdm

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
STOCK_FILE = DATA_DIR / "stock_data_features_v1.csv"
OUTPUT_FILE = DATA_DIR / "stock_sector_info.csv"

# ★手動補完リスト (yfinanceで取れない銘柄用)
MANUAL_PATCH = {
    "6026": {"Sector": "Communication Services", "Industry": "Advertising Agencies", "MarketCap": 0},  # GMO TECH
    "8279": {"Sector": "Consumer Defensive", "Industry": "Grocery Stores", "MarketCap": 0},  # ヤオコー
    "9164": {"Sector": "Industrials", "Industry": "Staffing & Employment Services", "MarketCap": 0},  # トライト
}


def main():
    print("--- 業種データ収集開始 (手動補完付き) ---")
    if not STOCK_FILE.exists():
        print("株価データが見つかりません")
        return

    # 銘柄リスト読み込み
    df = pd.read_csv(STOCK_FILE)
    if "Code" in df.columns:
        df.rename(columns={"Code": "code"}, inplace=True)
    codes = sorted(df["code"].astype(str).unique())

    sector_data = []
    print(f"対象: {len(codes)} 銘柄")

    for code in tqdm(codes):
        # 1. まず手動リストにあるかチェック
        if code in MANUAL_PATCH:
            info = MANUAL_PATCH[code]
            sector_data.append(
                {"code": code, "Sector": info["Sector"], "Industry": info["Industry"], "MarketCap": info["MarketCap"]}
            )
            continue

        # 2. なければ yfinance から取得
        try:
            ticker = yf.Ticker(f"{code}.T")
            info = ticker.info

            # セクター情報が取れたか確認
            sector = info.get("sector", "Unknown")
            if sector == "Unknown":
                # 取れなかった場合はエラー扱い
                raise ValueError("Sector not found")

            sector_data.append(
                {
                    "code": code,
                    "Sector": sector,
                    "Industry": info.get("industry", "Unknown"),
                    "MarketCap": info.get("marketCap", 0),
                }
            )
            time.sleep(0.5)  # 負荷軽減
        except Exception as e:
            print(f"警告: 銘柄コード {code} の情報取得に失敗 ({e})")
            # 本当に取れなかった場合
            sector_data.append({"code": code, "Sector": "Unknown", "Industry": "Unknown", "MarketCap": 0})

    # 保存
    df_sector = pd.DataFrame(sector_data)
    df_sector.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n完了: {OUTPUT_FILE}")

    # 確認表示
    print("\n--- 補完結果 ---")
    print(df_sector[df_sector["code"].isin(MANUAL_PATCH.keys())])


if __name__ == "__main__":
    main()
