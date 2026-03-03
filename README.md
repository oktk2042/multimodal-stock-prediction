# Multi-Modal Gated Transformer for Japanese Stock Return Prediction

Master's thesis research: Predicting 5-day-ahead stock returns for Japanese equities by fusing financial text (news headlines, EDINET filings) with market data through a gated cross-attention transformer.

## Project Structure

```
.
├── 1_data/                          # Data (not fully tracked in git)
│   ├── raw/                         #   Stock index constituent lists
│   ├── chABSA/                      #   Annotated financial sentiment data (FinBERT training)
│   ├── chABSA-dataset/              #   External dataset (git submodule)
│   ├── edinet_reports/              #   EDINET financial reports (large, gitignored)
│   ├── processed/                   #   Processed & integrated datasets
│   └── modeling_data/               #   Training-ready NPZ files (gitignored)
├── 2_src/                           # Source code
│   ├── data_collection/             #   Phase 1: Data acquisition (8 scripts)
│   ├── preprocessing/               #   Phase 2a: Basic feature engineering (4 scripts)
│   ├── feature_engineering/         #   Phase 2b: Sentiment & financial features (7 scripts)
│   ├── analysis/                    #   EDA & visualization (22 scripts)
│   └── models/                      #   Model training & evaluation
│       ├── arch/                    #     Model architectures (6 models)
│       ├── layers/                  #     Shared layers (RevIN, Attention, Embed)
│       ├── train_*.py               #     Training scripts (3 scripts)
│       └── *.py                     #     Evaluation & visualization (20+ scripts)
├── 3_reports/                       # Experiment results (figures, CSVs)
│   ├── final_consolidated_v2/       #   Final model comparison results
│   ├── analysis_output/             #   EDA figures
│   ├── final_figures*/              #   Paper-ready figures
│   └── best_case_studies/           #   Case study visualizations
├── 4_models/                        # Pre-trained FinBERT (gitignored)
├── pyproject.toml                   # Dependencies (uv)
├── Makefile                         # Pipeline automation
└── .env.example                     # Required API keys template
```

## Proposed Method: Multi-Modal Gated Transformer

The core contribution is **FusionTransformer** (`2_src/models/arch/fusion_transformer.py`), which combines:

- **RevIN** (Reversible Instance Normalization) for non-stationary time series
- **1D-CNN** for extracting local temporal patterns from market data
- **Gated Cross-Attention** for fusing text (FinBERT sentiment, financial metrics) with market data, where a sigmoid gate controls information flow
- **Positional Embedding** + Transformer Encoder for temporal modeling

The gate mechanism learns to **open** when text signals are informative and **close** when they are noise, improving robustness.

## Model Comparison

8 models are compared for 5-day return prediction on the top 200 Japanese stocks:

| Model | Accuracy (%) | Type |
|-------|-------------|------|
| **Multi-Modal Gated Transformer (Ours)** | **54.86** | Transformer (multi-modal) |
| Attention-LSTM | 54.52 | RNN |
| PatchTST | 54.10 | Transformer (patch-based) |
| DLinear | 50.99 | Linear (decomposition) |
| iTransformer | 48.15 | Transformer (inverted) |
| LightGBM | 46.52 | Gradient Boosting |
| Ridge Regression | 46.24 | Linear |
| Vanilla Transformer | 44.75 | Transformer |

### Ablation Study

| Variant | Accuracy (%) |
|---------|-------------|
| Proposed (Full) | 54.51 |
| w/o Text | 54.06 |
| w/o CNN | 53.83 |
| w/o Gating | 49.72 |

---

## Pipeline Execution Order

The research pipeline consists of 4 phases. Each script is designed to be run independently in the order shown below.

### Phase 1: Data Collection (`2_src/data_collection/`)

External APIs and data sources from which raw data is collected.

| # | Script | Purpose | Input | Output | API Required |
|---|--------|---------|-------|--------|-------------|
| 1-1 | `collect_all_constituents.py` | 6つの日本株指数（日経225, TOPIX Core30/100, JPX400, Growth等）の構成銘柄を取得 | CSV files, Web | `1_data/raw/nikkei_225.csv` etc. | - |
| 1-2 | `collect_master_stock_prices.py` | 全対象銘柄の日次OHLCVデータをyfinanceから取得 | Master stock list | `1_data/processed/all_stock_prices.csv` | - |
| 1-3 | `collect_edinet_complete.py` | EDINET APIから有価証券報告書・四半期報告書をZIPダウンロード（2018-2025） | Stock codes, EDINET API | `1_data/edinet_reports/01_zip_files/` | `EDINET_API_KEY` |
| 1-4 | `collect_historical_news.py` | Google News RSSから銘柄別の過去ニュース見出しを月次バッチで収集 | Stock data CSV | `1_data/processed/collected_news_historical_full.csv` | - |
| 1-5 | `collect_exchange_rate.py` | USD/JPY為替レートをyfinanceから取得 | - | `1_data/raw/usd_jpy_exchange_rate.csv` | - |
| 1-6 | `collect_global_macro.py` | グローバルマクロ指標（S&P500, NASDAQ, VIX, SOX, USD Index）を取得 | - | `1_data/processed/global_macro_features.csv` | - |
| 1-7 | `collect_sector_info.py` | 各銘柄のセクター・業種・時価総額をyfinanceから取得 | Stock list | `1_data/processed/stock_sector_info.csv` | - |
| 1-8 | `collect_tdnet_disclosures_v2.py` | J-Quants APIから適時開示情報を取得（2020-2025） | Search map JSON | `1_data/processed/tdnet_disclosures_v2.csv` | `JQUANTS_API_KEY` |

### Phase 2a: Preprocessing (`2_src/preprocessing/`)

Raw data from Phase 1 to basic features.

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 2-1 | `create_master_list.py` | 6指数の構成銘柄CSVを統合・重複除去してマスターリスト作成 | 6 constituent CSVs | `1_data/processed/master_stock_list.csv` |
| 2-2 | `generate_search_map.py` | J-Quants APIで企業名・セクター・EDINET コードを含む検索マップJSON生成 | Master list, J-Quants API | `1_data/processed/company_search_map.json` |
| 2-3 | `create_features.py` | 指数構成の要約作成、基本テクニカル指標（MA_5D/25D/75D）を付与 | Stock prices CSV | `1_data/processed/stock_data_with_technical_features.csv` |
| 2-4 | `add_advanced_technicals.py` | RSI, MACD, ボリンジャーバンドなど高度テクニカル指標を追加 | Technical features CSV | `1_data/processed/stock_data_with_all_technicals.csv` |

### Phase 2b: Feature Engineering (`2_src/feature_engineering/`)

Sentiment analysis, financial metric extraction, and dataset integration.

| # | Script | Purpose | Input | Output | Notes |
|---|--------|---------|-------|--------|-------|
| 2-5 | `train_finbert.py` | chABSAデータセットでBERTを金融センチメント分類にファインチューニング | `1_data/chABSA/data/annotated/*.json` | `4_models/finbert_chabsa_trained/` | GPU推奨 |
| 2-6 | `extract_features_from_large_csv.py` | 学習済みFinBERTでニュース見出しにセンチメントスコアを付与（チャンク処理） | News CSV | `1_data/processed/news_sentiment_historical.csv` | GPU推奨 |
| 2-7 | `extract_finbert_from_zips.py` | EDINET有報ZIPからテキスト抽出し、スライディングウィンドウでFinBERTスコア算出 | EDINET ZIPs | `1_data/processed/edinet_features_finbert_indices_strict.csv` | GPU推奨 |
| 2-8 | `extract_features_hybrid.py` | XBRLデータからルールベース＋Azure OpenAI LLMフォールバックで財務数値を抽出 | XBRL CSVs | `1_data/processed/edinet_features_financials_hybrid.csv` | `AZURE_OPENAI_*` |
| 2-9 | `extract_rss_features.py` | RSS見出しデータに対しセンチメントスコアを付与し、日次・銘柄レベルに集約 | Headlines CSV | `1_data/processed/news_sentiment_features.csv` | |
| 2-10 | `integrate_datasets_final_v5.py` | 株価・センチメント・財務・マクロ・セクター情報を時系列整合（merge_asof）で統合 | All processed CSVs | `1_data/processed/final_modeling_dataset_v5.csv` | |
| 2-11 | `make_dataset_for_training.py` | 5日先リターン予測のターゲット変数生成、テクニカル・ファンダメンタル特徴量のエンジニアリング | Top 200 dataset | `1_data/processed/dataset_for_modeling_top200_final.csv` | |

### Phase 3: Model Training (`2_src/models/`)

Hyperparameter optimization with Optuna and model comparison.

| # | Script | Purpose | Input | Output | Notes |
|---|--------|---------|-------|--------|-------|
| 3-1 | `train_ridge_optuna.py` | Ridge回帰のOptunaハイパーパラメータ最適化（50 trials）。線形ベースライン | Modeling CSV | `3_reports/*/predictions_Ridge.csv` | |
| 3-2 | `train_lgbm_optuna.py` | LightGBMのOptuna最適化（50 trials）。勾配ブースティングベースライン | Modeling CSV | `3_reports/*/predictions_LightGBM.csv` | |
| 3-3 | `train_deep_models_optuna.py` | 6種のDeep Learningモデル（LSTM, Transformer, DLinear, PatchTST, iTransformer, FusionTransformer）をOptuna最適化で学習。100エポック×30 trials | Modeling CSV | `3_reports/*/predictions_*.csv`, `best_model_*.pth` | GPU必須 |

### Phase 4: Evaluation & Analysis (`2_src/models/`)

| # | Script | Purpose | Input | Output |
|---|--------|---------|-------|--------|
| 4-1 | `run_ablation_study.py` | FusionTransformerの各コンポーネント（Text/Gate/CNN）の寄与度を検証。4変種を比較 | Modeling CSV | `ablation_study_results.csv`, chart |
| 4-2 | `compare_models.py` | 全8モデルの予測結果を集約し、RMSE/MAE/Accuracy/R2等のメトリクスで横断比較 | predictions CSVs | `final_model_comparison.csv`, bar charts |
| 4-3 | `consolidate_results.py` | 分散した学習結果を `final_consolidated_v2/` に統合し、一貫した命名規則で整理 | Multiple dirs | `3_reports/final_consolidated_v2/` |
| 4-4 | `perform_financial_analysis.py` | バックテスト・シャープレシオ・最大ドローダウンなどの金融パフォーマンス分析 | predictions CSVs | Financial summary CSV, charts |

### Model Architecture Files (`2_src/models/arch/`)

| File | Class | Description |
|------|-------|-------------|
| `fusion_transformer.py` | `FusionTransformer` | **提案手法**: マルチモーダル融合Transformer。RevIN + 1D-CNN + Gated Cross-Attention。テキスト情報と市場データをシグモイドゲートで動的に融合 |
| `transformer.py` | `VanillaTransformer` | 標準的なEncoder-only Transformer。Post-Norm、全時系列をフラット化して予測 |
| `itransformer.py` | `iTransformer` | 逆転Transformer。時間方向ではなく変数方向にAttentionを計算（変数間相関を学習） |
| `lstm_attn.py` | `AttentionLSTM` | LSTM + Bahdanau Attention。最終隠れ状態をQueryとし、全タイムステップにAttention |
| `patchtst.py` | `PatchTST` | パッチベースTransformer（Vision Transformer風）。時系列をパッチに分割しPosition Embedding |
| `dlinear.py` | `DLinear` | 分解ベース線形モデル。移動平均でトレンド/季節成分に分離し、それぞれ線形予測 |

### Shared Layers (`2_src/models/layers/`)

| File | Classes | Description |
|------|---------|-------------|
| `embed.py` | `PositionalEmbedding`, `DataEmbedding` | 正弦波位置エンコーディング + 値埋め込みの組み合わせ |
| `revin.py` | `RevIN` | Reversible Instance Normalization。時系列の非定常性に対応（ICLR 2022） |
| `self_attention_family.py` | `FullAttention`, `EncoderLayer`, `Encoder` | Scaled Dot-Product Attention、Pre/Post-Norm対応のEncoder Layer |

### Analysis & Visualization (`2_src/analysis/`)

| Script | Purpose |
|--------|---------|
| `analysis.py` | Top 200銘柄の指数構成比を集計 |
| `analysis_chABSA.py` | chABSAデータセットの極性分布を分析・可視化 |
| `analysis_chABSA_score.py` | chABSAの混合センチメント文（正と負を含む）の例を抽出 |
| `analysis_news_score.py` | ニュースセンチメントの正/負上位例を表示 |
| `analyze_data_distribution.py` | データ品質チェック：異常値検出、センチメント分布ヒストグラム |
| `analyze_news_details.py` | ニュース記事数上位20銘柄、メディアソース分布の可視化 |
| `analyze_news_sparce.py` | ニュースカバレッジの疎密分析（高/中/低頻度銘柄の比較） |
| `analyze_reset.py` | モデリングデータセットにターゲット変数（翌日ログリターン）を追加 |
| `analyze_text_stats.py` | ニュース（RSS）とEDINET（有報）のテキストデータ統計比較 |
| `create_appendix_table.py` | 論文付録用：200銘柄のLaTeX表（企業名、セクター、記事数）を生成 |
| `find_best_cases.py` | ケーススタディ抽出：ポジティブ/ネガティブ/ゲート閉じの3タイプ |
| `find_gate_closed_case.py` | ゲート機構がノイズをフィルタした特殊ケース（大幅値動き＋低ゲートスコア）を特定 |
| `generate_all_figures.py` | 論文用図表の一括生成マスタースクリプト |
| `generate_appendix_figures.py` | 付録用図表（予測プロット、散布図、特徴量重要度）の生成 |
| `generate_case_studies.py` | ケーススタディ可視化（株価チャート + ゲートスコア + ニュースイベント） |
| `generate_figures.py` | モデル比較チャート（精度、R2、RMSE）と予測時系列プロットの生成 |
| `generate_results_plots.py` | 補足的な比較プロット（全モデル精度比較、アーキテクチャ図） |
| `organize_edinet_files.py` | EDINET文書IDから銘柄コードへのマッピングとファイル整理 |
| `plot_news_distribution.py` | 銘柄あたりの月間ニュース記事数のヒストグラム |
| `process_edinet_zips.py` | EDINET ZIPからXBRL/HTMLの財務指標（売上高、営業利益等）を抽出 |
| `reshape_financial_data.py` | 抽出した財務データの標準化・ワイドフォーマット変換 |
| `run_comprehensive_eda.py` | 包括的EDA：ニュース分布、相関ヒートマップ、ニュースのボラティリティへの影響分析 |
| `select_top200_stocks_v2.py` | 流動性基準で上位200銘柄を選択（最低取引日数1000日以上） |

### Model Evaluation & Visualization (`2_src/models/`)

| Script | Purpose |
|--------|---------|
| `analyze_6stocks_performance.py` | 固定6銘柄（ニュース高/中/低頻度）のモデル別精度比較 |
| `analyze_data_distribution.py` | ニュース疎密度の検証、FinBERTスコアと5日リターンの相関分析 |
| `analyze_gate_behavior.py` | 6対象銘柄のゲート機構の挙動可視化（株価・ゲートスコア・ニュース数） |
| `analyze_stock_sector_top200.py` | Top 200銘柄のセクター構成ドーナツチャート |
| `calc_importance_only.py` | 学習済みモデルの置換型特徴量重要度を計算・可視化 |
| `find_best_cases.py` | ケーススタディ候補の抽出（センチメント vs リターンの乖離ケース） |
| `generate_10_case_studies.py` | Gate-Open 5例 + Gate-Closed 5例の実データケーススタディ可視化 |
| `generate_model_comparison_plots.py` | 8モデルの精度/R2/RMSEバーチャート生成 |
| `perform_financial_analysis.py` | バックテスト（方向精度、シャープレシオ、最大ドローダウン）の全モデル比較 |
| `plot_backtest.py` | ゲート戦略バックテスト：Gate Score > 0.25 の銘柄のみロング、累積リターン可視化 |
| `plot_backtest_2.py` | ロング/ショート戦略：全8モデルの上位5/下位5銘柄ポートフォリオ比較 |
| `plot_best_closed_case.py` | ゲート閉じケーススタディ（ノイズフィルタリング例）の可視化 |
| `plot_best_open_case.py` | ゲート開きケーススタディ（ニュース反応例）の可視化 |
| `plot_case_studies.py` | 汎用ケーススタディプロッター（任意の銘柄の株価+予測+ゲート） |
| `plot_news_distribution.py` | FinBERTセンチメントスコアの分布ヒストグラム |
| `plot_sector_comparison.py` | セクター別精度比較（電機、情報通信、医薬品等）の可視化 |
| `run_volatility_analysis.py` | ニュース有無によるボラティリティ・出来高への影響分析 |
| `visualize_gate_behavior.py` | 代表銘柄（SoftBank, Toyota, Sony等）のゲートスコア時系列可視化 |
| `prepare_for_modeling.py` | データセットをNumPy NPZ形式に変換し、Train/Val/Test分割 |
| `rename_models.py` | モデル名の正式名称への一括リネーム |

---

## Data Pipeline (Flow Diagram)

```
External APIs (yfinance, EDINET, Google RSS, J-Quants)
    │
    ▼
Phase 1: Data Collection (8 scripts) ──── 1_data/raw/, processed/
    │  collect_all_constituents.py → master stock list
    │  collect_master_stock_prices.py → daily OHLCV
    │  collect_edinet_complete.py → financial reports (ZIP)
    │  collect_historical_news.py → news headlines
    │  collect_exchange_rate.py → USD/JPY
    │  collect_global_macro.py → S&P500, VIX, etc.
    │  collect_sector_info.py → sector, market cap
    │  collect_tdnet_disclosures_v2.py → timely disclosures
    │
    ▼
Phase 2: Feature Engineering (11 scripts)
    │  create_master_list.py → unified stock list
    │  generate_search_map.py → company metadata JSON
    │  create_features.py → basic technical (MA)
    │  add_advanced_technicals.py → RSI, MACD, BB
    │  train_finbert.py → fine-tuned FinBERT model
    │  extract_features_from_large_csv.py → news sentiment
    │  extract_finbert_from_zips.py → EDINET sentiment
    │  extract_features_hybrid.py → financial metrics (XBRL+LLM)
    │  integrate_datasets_final_v5.py → merge all sources
    │  select_top200_stocks_v2.py → top 200 by liquidity
    │  make_dataset_for_training.py → final features + targets
    │
    ▼
Phase 3: Model Training (Optuna HPO)
    │  train_ridge_optuna.py → Ridge regression baseline
    │  train_lgbm_optuna.py → LightGBM baseline
    │  train_deep_models_optuna.py → 6 deep learning models
    │
    ▼
Phase 4: Evaluation ──────────────────── 3_reports/
    │  run_ablation_study.py → component analysis
    │  compare_models.py → cross-model metrics
    │  consolidate_results.py → unified output
    │  perform_financial_analysis.py → backtesting
    │
    ▼
Analysis & Visualization (22+ scripts)
       generate_all_figures.py → paper-ready figures
       generate_case_studies.py → gate behavior examples
       plot_backtest.py → portfolio simulation
       ... (see table above for full list)
```

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip
- NVIDIA GPU with CUDA 12.1 (for training)

### Installation

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd multimodal-stock-prediction

# Install dependencies
uv sync

# Copy and fill in API keys
cp .env.example .env
```

### Required API Keys

| Key | Service | Purpose |
|-----|---------|---------|
| `EDINET_API_KEY` | [EDINET](https://disclosure2.edinet-fsa.go.jp/) | Financial report downloads |
| `JQUANTS_API_KEY` | [J-Quants](https://jpx-jquants.com/) | Company metadata, timely disclosures |
| `GOOGLE_API_KEY` | Google | News search |
| `NEWS_API_KEY` | [NewsAPI](https://newsapi.org/) | News headlines |
| `AZURE_OPENAI_*` | Azure OpenAI | Hybrid financial metric extraction |

## Usage

```bash
# Show all available commands
make help

# Run full pipeline (data collection -> training -> evaluation)
make all

# Or run individual phases:
make collect          # Phase 1: Data collection
make features         # Phase 2: Preprocessing & feature engineering
make train            # Phase 3: Model training (Ridge -> LightGBM -> Deep)
make evaluate         # Phase 4: Evaluation & analysis

# Code quality
make lint             # Run ruff linter
make format           # Auto-format with ruff
make clean            # Remove __pycache__ files
```

### Running Individual Scripts

Each script can also be run independently:

```bash
# Example: Train only deep learning models
cd 2_src/models
uv run python train_deep_models_optuna.py

# Example: Generate analysis figures
uv run python 2_src/analysis/generate_all_figures.py

# Example: Run ablation study
cd 2_src/models
uv run python run_ablation_study.py
```

## Key Technologies

- **PyTorch** + Transformers for deep learning
- **Optuna** for hyperparameter optimization
- **FinBERT** (fine-tuned on [chABSA](https://github.com/chakki-works/chABSA-dataset)) for Japanese financial sentiment
- **LightGBM** / **scikit-learn** for baseline models
- **yfinance** / **EDINET API** / **J-Quants API** for data acquisition
- **Azure OpenAI** for hybrid XBRL extraction (rule-based + LLM fallback)

## Output

Final results are stored in `3_reports/final_consolidated_v2/`:

- `model_comparison_summary.csv` - All 8 models' metrics (RMSE, MAE, Accuracy, R2)
- `predictions_*.csv` - Per-model prediction results (actual vs predicted)
- `*_feature_importance.csv` / `.png` - Feature importance rankings
- `*_learning_curve.png` - Training/validation loss curves
- `ablation_study_results.csv` / `.png` - Component contribution analysis
- `final_*_comparison.png` - Paper-ready comparison charts

## License

MIT
