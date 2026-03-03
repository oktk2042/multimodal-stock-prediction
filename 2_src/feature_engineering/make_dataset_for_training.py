from pathlib import Path

import numpy as np
import pandas as pd

# ==========================================
# 1. 設定
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "1_data" / "processed"

# 入力: 選定されたTop200銘柄のデータ
INPUT_FILE = DATA_DIR / "final_data_top200.csv"

# 出力: モデル学習用データセット
OUTPUT_FILE = DATA_DIR / "dataset_for_modeling_top200.csv"


# ==========================================
# 2. 特徴量エンジニアリング関数
# ==========================================
def engineer_features(df):
    """
    Top200銘柄データに対して、モデル学習用の特徴量を生成する
    """
    df = df.copy()

    # 日付ソート（必須）
    df = df.sort_values(["code", "Date"])

    print("特徴量生成プロセスを開始します...")

    # ------------------------------------------------
    # A. ターゲット（予測対象）の作成: 5日後予測
    # ------------------------------------------------
    # shift(-5) で「5行先（＝5営業日後）」のデータを取得
    # 目的変数1: 5日後のリターン (学習用)
    df["Target_Return_5D"] = df.groupby("code")["Close"].shift(-5) / df["Close"] - 1.0

    # 目的変数2: 5日後の株価 (評価用・RMSE計算用)
    df["Target_Close_5D"] = df.groupby("code")["Close"].shift(-5)

    # ------------------------------------------------
    # B. テクニカル特徴量 (時系列情報)
    # ------------------------------------------------
    # 1. 対数収益率 (Log Return): 正規分布に近づける
    df["Log_Return"] = np.log(df["Close"] / df.groupby("code")["Close"].shift(1))

    # 2. 過去5日間の累積リターン (Momentum)
    df["Return_5D"] = df["Close"] / df.groupby("code")["Close"].shift(5) - 1.0

    # 3. ボラティリティ (20日間の標準偏差)
    df["Volatility_20D"] = df.groupby("code")["Log_Return"].transform(lambda x: x.rolling(20).std())

    # 4. 移動平均乖離率 (25日)
    if "MA_25D" in df.columns:
        df["MA_Gap_25D"] = df["Close"] / df["MA_25D"] - 1.0

    # 5. ラグ特徴量 (Phase 1 LightGBM用)
    # Deep Learningでは不要だが、LightGBM/Ridgeは「行」で判断するため、過去の値を横に並べる
    for lag in [1, 2, 3, 4, 5]:
        df[f"Log_Return_Lag{lag}"] = df.groupby("code")["Log_Return"].shift(lag)

    # ------------------------------------------------
    # C. 財務特徴量 (対数化・比率化)
    # ------------------------------------------------
    # 売上高の対数化 (0の場合は0のままにするため log1p を使用)
    if "NetSales" in df.columns:
        df["Log_NetSales"] = np.log1p(df["NetSales"])

    # 営業利益率 (Operating Margin)
    # 売上が0の場合はNaNになる -> 0で埋める
    if "OperatingIncome" in df.columns and "NetSales" in df.columns:
        df["Operating_Margin"] = df["OperatingIncome"] / df["NetSales"].replace(0, np.nan)
        df["Operating_Margin"] = df["Operating_Margin"].fillna(0)

    # 売上高対時価総額比率 (PSRの逆数)
    if "MarketCap" in df.columns and "NetSales" in df.columns:
        df["Sales_to_MarketCap"] = df["NetSales"] / df["MarketCap"].replace(0, np.nan)
        df["Sales_to_MarketCap"] = df["Sales_to_MarketCap"].fillna(0)

    # ------------------------------------------------
    # D. マクロ・センチメント特徴量
    # ------------------------------------------------
    # マクロ指数の変化率
    macro_cols = ["SP500_Close", "NASDAQ_Close", "USD_Index_Close", "SOX_Close"]
    for col in macro_cols:
        if col in df.columns:
            df[f"{col}_Change"] = df[col].pct_change()

    # VIXは差分をとる
    if "VIX_Close" in df.columns:
        df["VIX_Change"] = df["VIX_Close"].diff()

    # ------------------------------------------------
    # E. クリーニング & 保存準備
    # ------------------------------------------------
    # 無限大(inf)を除去
    df = df.replace([np.inf, -np.inf], np.nan)

    # ターゲット生成(shift -5)やラグ生成(shift +5)で発生したNaN行を削除
    # これを行わないと、学習時にNaNエラーが出る
    before_drop = len(df)
    df = df.dropna(subset=["Target_Return_5D", "Log_Return", "Volatility_20D", "Log_Return_Lag5"])
    print(f"NaN除去: {before_drop:,} -> {len(df):,} 行 ({(before_drop - len(df)):,} 行削除)")

    return df


def main():
    print("--- 学習用データセット作成 (Feature Engineering) ---")

    if not INPUT_FILE.exists():
        print(f"エラー: {INPUT_FILE} が見つかりません。先にTop200選定を行ってください。")
        return

    # 1. データ読み込み
    print(f"データを読み込み中: {INPUT_FILE.name}")
    df = pd.read_csv(INPUT_FILE, dtype={"code": str}, low_memory=False, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"])

    print(f"入力データ: {len(df):,} 行, {df['code'].nunique()} 銘柄")

    # 2. 特徴量生成
    df_engineered = engineer_features(df)

    # 3. 保存
    print(f"保存中... ({len(df_engineered):,} 行)")
    df_engineered.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("\n✅ 完了！")
    print(f"出力ファイル: {OUTPUT_FILE}")
    print("\n[作成された主な特徴量]")
    print(" - Target_Return_5D: 5日後のリターン (目的変数)")
    print(" - Target_Close_5D:  5日後の株価 (評価用)")
    print(" - Log_Return:       対数収益率")
    print(" - Log_Return_Lag*:  ラグ特徴量 (1~5日前)")
    print(" - Operating_Margin: 営業利益率")


if __name__ == "__main__":
    main()
