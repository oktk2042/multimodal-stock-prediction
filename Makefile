.PHONY: setup lint format test clean collect preprocess features train evaluate analysis all help

PYTHON = uv run python
SRC    = 2_src

# ============================================================
# Setup
# ============================================================

setup: ## Install dependencies and pre-commit hooks
	uv sync
	uv run pre-commit install || true

# ============================================================
# Code Quality
# ============================================================

lint: ## Run linter (ruff check)
	uv run ruff check $(SRC)/

format: ## Auto-format code (ruff format)
	uv run ruff format $(SRC)/

test: ## Run tests
	uv run pytest tests/ -v

# ============================================================
# Phase 1: Data Collection
# ============================================================

collect-constituents: ## Collect stock index constituents
	$(PYTHON) $(SRC)/data_collection/collect_all_constituents.py

collect-master-list: ## Create master stock list
	$(PYTHON) $(SRC)/preprocessing/create_master_list.py

collect-search-map: ## Generate company search map (requires J-Quants API)
	$(PYTHON) $(SRC)/preprocessing/generate_search_map.py

collect-prices: ## Fetch stock prices from yfinance
	$(PYTHON) $(SRC)/data_collection/collect_master_stock_prices.py

collect-edinet: ## Download EDINET financial reports (requires EDINET API key)
	$(PYTHON) $(SRC)/data_collection/collect_edinet_complete.py

collect-news: ## Collect historical news from Google RSS
	$(PYTHON) $(SRC)/data_collection/collect_historical_news.py

collect-macro: ## Fetch global macro indicators
	$(PYTHON) $(SRC)/data_collection/collect_global_macro.py
	$(PYTHON) $(SRC)/data_collection/collect_exchange_rate.py

collect-sector: ## Fetch sector information
	$(PYTHON) $(SRC)/data_collection/collect_sector_info.py

collect: collect-constituents collect-master-list collect-search-map collect-prices collect-edinet collect-news collect-macro collect-sector ## Run all data collection

# ============================================================
# Phase 2: Preprocessing & Feature Engineering
# ============================================================

preprocess: ## Create basic technical features
	$(PYTHON) $(SRC)/preprocessing/create_features.py
	$(PYTHON) $(SRC)/preprocessing/add_advanced_technicals.py

train-finbert: ## Fine-tune FinBERT on chABSA dataset (requires GPU)
	$(PYTHON) $(SRC)/feature_engineering/train_finbert.py

extract-sentiment: ## Extract sentiment scores from news and reports
	$(PYTHON) $(SRC)/feature_engineering/extract_features_from_large_csv.py
	$(PYTHON) $(SRC)/feature_engineering/extract_finbert_from_zips.py

extract-financials: ## Extract financial metrics from EDINET XBRL (requires Azure OpenAI)
	$(PYTHON) $(SRC)/feature_engineering/extract_features_hybrid.py

integrate: ## Integrate all datasets into unified modeling dataset
	$(PYTHON) $(SRC)/feature_engineering/integrate_datasets_final_v5.py

select-stocks: ## Select top 200 stocks by liquidity
	$(PYTHON) $(SRC)/analysis/select_top200_stocks_v2.py

make-dataset: ## Create final training dataset with engineered features
	$(PYTHON) $(SRC)/feature_engineering/make_dataset_for_training.py

features: preprocess extract-sentiment extract-financials integrate select-stocks make-dataset ## Run all feature engineering

# ============================================================
# Phase 3: Model Training
# ============================================================

train-ridge: ## Train Ridge regression baseline (Optuna)
	cd $(SRC)/models && $(PYTHON) train_ridge_optuna.py

train-lgbm: ## Train LightGBM (Optuna)
	cd $(SRC)/models && $(PYTHON) train_lgbm_optuna.py

train-deep: ## Train all deep learning models (Optuna, requires GPU)
	cd $(SRC)/models && $(PYTHON) train_deep_models_optuna.py

train: train-ridge train-lgbm train-deep ## Train all models

# ============================================================
# Phase 4: Evaluation & Analysis
# ============================================================

ablation: ## Run ablation study for FusionTransformer
	cd $(SRC)/models && $(PYTHON) run_ablation_study.py

compare: ## Compare all models
	cd $(SRC)/models && $(PYTHON) compare_models.py

consolidate: ## Consolidate results into final directory
	cd $(SRC)/models && $(PYTHON) consolidate_results.py

evaluate: ablation compare consolidate ## Run all evaluation

analysis: ## Generate analysis figures and tables
	$(PYTHON) $(SRC)/analysis/generate_all_figures.py

# ============================================================
# Full Pipeline
# ============================================================

all: collect features train evaluate analysis ## Run entire pipeline end-to-end

# ============================================================
# Utilities
# ============================================================

clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
