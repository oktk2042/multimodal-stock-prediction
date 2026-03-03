from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

# ==========================================
# 保存設定
# ==========================================
# 論文の画像フォルダに合わせてパスを設定
OUTPUT_DIR = Path("C:/M2_Research_Project/3_reports/final_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 日本語フォント設定 (環境に合わせて変更してください)
import matplotlib

matplotlib.rcParams["font.family"] = "MS Gothic"


def draw_box(ax, x, y, w, h, text, color="#EFEFEF", edge="black", fontsize=10, fontcolor="black"):
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.1", linewidth=1.5, edgecolor=edge, facecolor=color
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, fontweight="bold", color=fontcolor)
    return x + w / 2, y + h, x + w / 2, y  # Top, Bottom coordinates


def draw_arrow(ax, x1, y1, x2, y2, text=None):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    if text:
        ax.text(
            (x1 + x2) / 2,
            (y1 + y2) / 2,
            text,
            ha="center",
            va="center",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
        )


# ==========================================
# 1. 全体アーキテクチャ図 (model_architecture.png)
# ==========================================
def generate_model_architecture():
    fig, ax = plt.subplots(figsize=(10, 12))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis("off")

    # --- Inputs ---
    draw_box(ax, 1, 12.5, 3, 0.8, "数値データ\n(株価・指標)", color="#E3F2FD")
    draw_box(ax, 6, 12.5, 3, 0.8, "テキストデータ\n(ニュース)", color="#FFF3E0")

    # --- Preprocessing ---
    # Market Path
    draw_arrow(ax, 2.5, 12.5, 2.5, 11.5)
    draw_box(ax, 1, 10.5, 3, 1.0, "RevIN\n(分布正規化)", color="#BBDEFB")

    draw_arrow(ax, 2.5, 10.5, 2.5, 9.5)
    draw_box(ax, 1, 8.5, 3, 1.0, "1D-CNN\n(局所トレンド抽出)", color="#90CAF9")

    # Text Path
    draw_arrow(ax, 7.5, 12.5, 7.5, 11.5)
    draw_box(ax, 6, 10.5, 3, 1.0, "LLM (GPT-4o)\n(情報抽出)", color="#FFE0B2")

    draw_arrow(ax, 7.5, 10.5, 7.5, 9.5)
    draw_box(ax, 6, 8.5, 3, 1.0, "Linear Projection\n+ Positional Enc.", color="#FFCC80")

    # --- Fusion ---
    draw_arrow(ax, 2.5, 8.5, 4, 7.5)  # Market to Fusion
    draw_arrow(ax, 7.5, 8.5, 6, 7.5)  # Text to Fusion

    draw_box(
        ax, 3, 6.5, 4, 1.0, "Gated Cross-Attention\n(動的統合)", color="#D1C4E9", edge="#512DA8", fontcolor="black"
    )

    # --- Temporal Modeling ---
    draw_arrow(ax, 5, 6.5, 5, 5.5)
    draw_box(ax, 3, 4.5, 4, 1.0, "Transformer Encoder\n(時系列学習)", color="#C5CAE9")

    # --- Prediction ---
    draw_arrow(ax, 5, 4.5, 5, 3.5)
    draw_box(ax, 3.5, 2.5, 3, 0.8, "Prediction Head\n(Flatten + Linear)", color="#E0E0E0")

    draw_arrow(ax, 5, 2.5, 5, 1.5)
    draw_box(ax, 3.5, 0.5, 3, 1.0, "RevIN Inverse\n(逆変換・出力)", color="#BBDEFB")

    plt.title("Multi-Modal Gated Transformer Architecture", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_architecture.png", dpi=300)
    print("保存完了: model_architecture.png")


# ==========================================
# 2. Gate機構の詳細図 (gate_mechanism_detail.png)
# ==========================================
def generate_gate_detail():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Inputs
    draw_box(ax, 1, 6.5, 2.5, 0.8, "Market (数値)\n[Query]", color="#90CAF9")
    draw_box(ax, 6.5, 6.5, 2.5, 0.8, "Text (ニュース)\n[Key/Value]", color="#FFCC80")

    # Attention Flow
    draw_arrow(ax, 2.25, 6.5, 4, 5.5)
    draw_arrow(ax, 7.75, 6.5, 6, 5.5)
    draw_box(ax, 4, 4.5, 2, 1.0, "Cross-Attention\n(Softmax)", color="#E1BEE7")

    # Gate Generation Flow (重要: Marketから分岐)
    draw_arrow(ax, 2.25, 6.5, 2.25, 4.5)
    draw_box(ax, 1, 3.5, 2.5, 1.0, "Gate Generator\nσ(Linear)", color="#CE93D8", edge="#4A148C")

    # Element-wise Product
    draw_arrow(ax, 5, 4.5, 5, 3.5)  # Attn output
    draw_arrow(ax, 3.5, 3.5, 4.5, 2.5)  # Gate output

    ax.text(5, 2.5, "⊗", fontsize=20, ha="center", va="center")  # Multiplication symbol
    ax.text(5.8, 2.5, "重み付け\n(Gating)", fontsize=9, ha="left", va="center")

    # Residual Connection
    draw_arrow(ax, 5, 2.2, 5, 1.5)  # Fused output

    # Skip connection (Original Market info)
    ax.plot([1.5, 1.5, 3, 3], [6.5, 1.0, 1.0, 1.0], color="#555555", linestyle="--", zorder=0)
    draw_arrow(ax, 3, 1.0, 3.8, 1.0)

    draw_box(ax, 3.8, 0.5, 2.4, 0.8, "Add & Norm\n(残差結合)", color="#D1C4E9")

    draw_arrow(ax, 5, 0.5, 5, 0.0)
    ax.text(5, -0.2, "To Transformer Encoder", ha="center", va="top", fontweight="bold")

    plt.title("Gated Cross-Attention Mechanism (詳細)", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gate_mechanism_detail.png", dpi=300)
    print("保存完了: gate_mechanism_detail.png")


if __name__ == "__main__":
    generate_model_architecture()
    generate_gate_detail()
