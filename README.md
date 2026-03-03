# Multi-Modal Gated Transformer for Japanese Stock Return Prediction

Master's thesis research: Predicting 5-day-ahead stock returns for Japanese equities by fusing financial text (news headlines, EDINET filings) with market data through a gated cross-attention transformer.

## Project Structure

```
.
├── 1_data/                          # Data (not fully tracked in git)
│   ├── raw/                         #   Stock index constituent lists
│   ├── chABSA/                      #   Annotated financial sentiment data
│   ├── edinet_reports/              #   EDINET financial reports (large)
│   ├── processed/                   #   Processed & integrated datasets
│   └── modeling_data/               #   Training-ready NPZ files
├── 2_src/                           # Source code
│   ├── data_collection/             #   Phase 1: Data acquisition (8 scripts)
│   ├── preprocessing/               #   Phase 2: Basic feature engineering
│   ├── feature_engineering/         #   Phase 2: Sentiment & financial features
│   ├── analysis/                    #   EDA & visualization
│   └── models/                      #   Model training & evaluation
│       ├── arch/                    #     Model architectures (6 models)
│       └── layers/                  #     Shared layers (RevIN, Attention, Embed)
├── 3_reports/                       # Experiment results (figures, CSVs)
├── 4_models/                        # Pre-trained FinBERT (not tracked in git)
├── pyproject.toml                   # Dependencies (uv)
├── Makefile                         # Pipeline automation
└── .env.example                     # Required API keys template
```

## Proposed Method: Multi-Modal Gated Transformer

The core contribution is **FusionTransformer**, which combines:

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
| `JQUANTS_API_KEY` | [J-Quants](https://jpx-jquants.com/) | Company metadata, disclosures |
| `GOOGLE_API_KEY` | Google | News search |
| `NEWS_API_KEY` | [NewsAPI](https://newsapi.org/) | News headlines |
| `AZURE_OPENAI_*` | Azure OpenAI | Hybrid financial extraction |

## Usage

```bash
# Show all available commands
make help

# Run full pipeline (data collection → training → evaluation)
make all

# Or run individual phases:
make collect          # Phase 1: Data collection
make features         # Phase 2: Preprocessing & feature engineering
make train            # Phase 3: Model training (Ridge → LightGBM → Deep)
make evaluate         # Phase 4: Evaluation & analysis

# Code quality
make lint             # Run ruff linter
make format           # Auto-format with ruff
```

### Individual Scripts

Each script can also be run independently:

```bash
# Example: Train only deep learning models
cd 2_src/models
uv run python train_deep_models_optuna.py

# Example: Generate analysis figures
uv run python 2_src/analysis/generate_all_figures.py
```

## Data Pipeline

```
External APIs (yfinance, EDINET, Google RSS, J-Quants)
    │
    ▼
Phase 1: Data Collection ─────────── 1_data/raw/, processed/
    │
    ▼
Phase 2: Feature Engineering
    ├── FinBERT fine-tuning (chABSA) ─ 4_models/finbert_chabsa_trained/
    ├── Sentiment extraction ───────── News & EDINET FinBERT scores
    ├── Financial metrics (XBRL+LLM) ─ NetSales, OperatingIncome
    ├── Technical indicators ───────── RSI, MACD, Bollinger Bands
    └── Dataset integration ────────── 1_data/processed/final_data_top200.csv
    │
    ▼
Phase 3: Model Training (Optuna HPO)
    ├── Ridge / LightGBM (baselines)
    └── 6 Deep Models (LSTM, Transformers, FusionTransformer)
    │
    ▼
Phase 4: Evaluation ────────────────── 3_reports/
    ├── Model comparison
    ├── Ablation study
    ├── Backtesting
    └── Case studies (gate behavior)
```

## Key Technologies

- **PyTorch** + Transformers for deep learning
- **Optuna** for hyperparameter optimization
- **FinBERT** (fine-tuned on chABSA) for Japanese financial sentiment
- **LightGBM** / **scikit-learn** for baseline models
- **yfinance** / **EDINET API** / **J-Quants API** for data acquisition
- **Azure OpenAI** for hybrid XBRL extraction

## License

MIT
