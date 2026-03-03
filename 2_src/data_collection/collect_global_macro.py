from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"
OUTPUT_FILE = DATA_DIR / "global_macro_features.csv"

INDICES = {"^GSPC": "SP500", "^IXIC": "NASDAQ", "^VIX": "VIX", "^SOX": "SOX", "DX-Y.NYB": "USD_Index"}


def main():
    print("--- グローバル・マクロ指標収集 ---")
    dfs = []
    for ticker, name in INDICES.items():
        print(f"取得中: {name}...")
        try:
            data = yf.download(ticker, start="2018-01-01", progress=False)
            df = data[["Close"]].copy()
            df.columns = [f"{name}_Close"]
            dfs.append(df)
        except Exception as e:
            print(f"エラー {name}: {e}")

    if dfs:
        df_macro = pd.concat(dfs, axis=1)
        df_macro.index.name = "Date"
        df_macro = df_macro.reset_index()
        df_macro["Date"] = pd.to_datetime(df_macro["Date"]).dt.normalize()
        df_macro = df_macro.ffill()  # 休日埋め
        df_macro.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print("完了")


if __name__ == "__main__":
    main()
