from pathlib import Path

import pandas as pd

# ==========================================
# 設定: 予測結果ファイル（Gate Score入り）を指定
# ==========================================
PROJECT_ROOT = Path(".").resolve()
PRED_FILE = PROJECT_ROOT / "3_reports" / "final_consolidated_v2" / "predictions_MultiModalGatedTransformer.csv"
OUTPUT_DIR = PROJECT_ROOT / "3_reports" / "phase3_production_strict"


def find_gate_closed_case():
    print(f"読み込み中: {PRED_FILE}")
    if not PRED_FILE.exists():
        print("予測結果ファイルが見つかりません。パスを確認してください。")
        return

    df = pd.read_csv(PRED_FILE)

    # 実際の5日後リターンを計算 (Target_Close_5D / Close - 1)
    df["Actual_Return_5D"] = df["Target_Close_5D"] / df["Close"] - 1

    # --- 条件: ノイズ遮断 (Gate Closed) ---
    # 1. 株価は大きく動いた (リターン絶対値 > 5%)
    # 2. しかし Gate は閉じていた (Gate Score < 0.25)
    #    → 「ニュース材料ではなく、需給や地合いで動いた」と判断したケース
    closed_cases = df[(df["Gate_Score"] < 0.25) & (df["Actual_Return_5D"].abs() > 0.05)].sort_values(
        "Gate_Score", ascending=True
    )  # Gateが固く閉じている順

    print("\n=== Gate Closed Cases (Top 5) ===")
    cols = ["Date", "code", "Name", "Gate_Score", "Actual_Return_5D"]
    print(df[cols].head(5).to_string(index=False))

    # 保存
    save_path = OUTPUT_DIR / "case_study_gate_closed_final.csv"
    closed_cases.head(20).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved to: {save_path}")


if __name__ == "__main__":
    find_gate_closed_case()
