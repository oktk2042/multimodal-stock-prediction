from pathlib import Path

import pandas as pd

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力ファイル\
INPUT_FILE = DATA_DIR / "final_modeling_dataset_v5.csv"

# 出力ファイル
OUTPUT_FILE = DATA_DIR / "dataset_for_modeling_top200_final.csv"
LIST_OUTPUT_FILE = DATA_DIR / "top200_stock_list_final.csv"

# フィルタ条件
TARGET_START_YEAR = 2018  # 分析期間の開始
TARGET_END_YEAR = 2025  # 分析期間の終了
MIN_DAYS_REQUIRED = 1000  # 期間中に最低限必要なデータ数 (上場廃止や新規上場を除外)
TOP_N = 200  # 選定する銘柄数


def main():
    print("--- Top 200 銘柄選定プロセス開始 ---")

    # 1. データの読み込み
    if not INPUT_FILE.exists():
        print(f"❌ エラー: 入力ファイルが見つかりません {INPUT_FILE}")
        return

    print(f"1. データを読み込み中... ({INPUT_FILE.name})")
    # Codeは文字列として読み込む
    df = pd.read_csv(INPUT_FILE, dtype={"Code": str}, low_memory=False)
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"   -> 全データ数: {len(df):,} 行")
    print(f"   -> 全銘柄数: {df['Code'].nunique()} 社")

    # 2. 期間フィルタ & データ品質チェック
    print("\n2. データ品質フィルタ実行中...")

    # 指定期間内のデータのみ抽出
    mask_period = (df["Date"].dt.year >= TARGET_START_YEAR) & (df["Date"].dt.year <= TARGET_END_YEAR)
    df_period = df[mask_period].copy()

    # 各銘柄のデータ日数をカウント
    stock_counts = df_period.groupby("Code").size()

    # 日数が足りている銘柄のみ残す
    valid_codes = stock_counts[stock_counts >= MIN_DAYS_REQUIRED].index.tolist()

    print(f"   -> 期間: {TARGET_START_YEAR} ~ {TARGET_END_YEAR}")
    print(f"   -> {MIN_DAYS_REQUIRED}日以上のデータがある銘柄: {len(valid_codes)} / {df['Code'].nunique()}")

    if len(valid_codes) < TOP_N:
        print("⚠️ 警告: 有効な銘柄数が目標(200)を下回っています。条件(MIN_DAYS_REQUIRED)を緩和してください。")

    # 3. 流動性フィルタ (売買代金)
    print(f"\n3. 流動性スコア計算 (平均売買代金 Top {TOP_N})...")

    # フィルタ通過銘柄のみに絞る
    df_valid = df_period[df_period["Code"].isin(valid_codes)].copy()

    # 売買代金 (Trading Value) = 終値 * 出来高
    # ※ 欠損値がある場合は0埋めして計算
    df_valid["TradingValue"] = df_valid["Close"].fillna(0) * df_valid["Volume"].fillna(0)

    # 銘柄ごとの「平均売買代金」を算出
    avg_trading_value = df_valid.groupby("Code")["TradingValue"].mean()

    # 上位N銘柄を選定
    top_codes = avg_trading_value.sort_values(ascending=False).head(TOP_N).index.tolist()

    # 4. 結果の確認と抽出
    print(f"\n--- 選定された Top {len(top_codes)} 銘柄 (一部) ---")
    # 銘柄名があれば表示したいが、現在のデータセットにNameが含まれているか確認
    if "Name" in df_valid.columns:
        name_map = df_valid.drop_duplicates("Code").set_index("Code")["Name"]
        for i, code in enumerate(top_codes[:5]):
            print(f" {i + 1}. {code}: {name_map.get(code, 'Unknown')} (平均売買代金: {avg_trading_value[code]:,.0f})")
    else:
        for i, code in enumerate(top_codes[:5]):
            print(f" {i + 1}. {code} (平均売買代金: {avg_trading_value[code]:,.0f})")

    # 5. 最終データセットの作成
    print("\n5. 最終データセットを保存中...")

    # 選定銘柄の全期間データを抽出 (期間フィルタ前の df から抽出して、全期間データを保持する)
    df_final = df[df["Code"].isin(top_codes)].copy()

    # ソート
    df_final.sort_values(["Code", "Date"], inplace=True)

    # 保存
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"   -> データセット保存完了: {OUTPUT_FILE}")
    print(f"   -> 形状: {df_final.shape}")

    # リストの保存
    pd.Series(top_codes, name="Code").to_csv(LIST_OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"   -> 銘柄リスト保存完了: {LIST_OUTPUT_FILE}")

    print("\n✅ Top 200 銘柄の選定と抽出が完了しました。")


if __name__ == "__main__":
    main()
