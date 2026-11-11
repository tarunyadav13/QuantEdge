# QuantEdge — ML-Powered Quantitative Research & Portfolio Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Project-Research%20Prototype-informational)]()
[![Market](https://img.shields.io/badge/Market-Indian%20Equities-orange)]()

QuantEdge is a quantitative research platform for Indian equities. It brings together market-data processing, technical-indicator feature engineering, machine-learning experiments, systematic trading strategies, realistic backtesting, portfolio analysis, and explainability in one research workflow.

The project is designed to **test hypotheses rather than assume that a complex model will outperform a simple strategy**. Experiments are evaluated using chronological data splits, execution delays, transaction costs, and risk-adjusted metrics. Results shown in the dashboard are loaded from saved experiment outputs; the dashboard does not retrain models when it opens.

> **Research disclaimer:** QuantEdge is an educational research prototype, not investment advice or a live trading system. Historical results do not guarantee future performance. Data-provider availability, survivorship bias, corporate actions, liquidity, taxes, market impact, and execution constraints can affect results.

---

## 1. Problem Statement

Stock-market research often involves disconnected notebooks and scripts for data collection, indicators, prediction, strategy testing, and portfolio evaluation. A model can also appear successful when the evaluation contains look-ahead leakage, unrealistic execution assumptions, or no transaction costs.

QuantEdge addresses this by providing a repeatable research workflow that:

- Studies multiple Indian equities using a consistent data and feature pipeline.
- Compares traditional technical strategies with machine-learning and hybrid approaches.
- Separates training, validation, and final testing chronologically.
- Evaluates trading outcomes after transaction costs and slippage.
- Compares strategy returns with buy-and-hold and benchmark proxies.
- Reports both predictive quality and portfolio/trading performance.
- Makes experiment outputs inspectable through a Streamlit research dashboard.

The current research universe includes **RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, and SBIN**. **NIFTYBEES** is used as a market/benchmark proxy where the experiment requires benchmark features or comparison.

## 2. System Architecture

```text
                 Indian Equity Market Data
                           |
                           v
                 Data Ingestion / Storage
                    data/raw/*.csv
                           |
                           v
                 Data Cleaning & Alignment
                           |
                           v
                  Feature Engineering
          Returns | Trend | Momentum | Volatility
             Volume | Benchmark-relative features
                           |
             +-------------+-------------+
             |             |             |
             v             v             v
       Traditional      ML Models     Sequence Models
       Strategies       RF / HGB      LSTM / GRU
       RSI / MACD       XGBoost       (where available)
       SMA / BB         LightGBM
             |             |             |
             +-------------+-------------+
                           |
                           v
                Chronological Evaluation
             Train -> Validation -> Test
                  Purging / Thresholding
                           |
                           v
                  Strategy Backtesting
          Signal delay | Costs | Slippage | Equity
                           |
              +------------+------------+
              |            |            |
              v            v            v
          Benchmark     Portfolio    Explainability
          Comparison     & Risk       SHAP outputs
              |            |            |
              +------------+------------+
                           |
                           v
                 Saved Research Artifacts
                           |
                           v
                  Streamlit Dashboard
```

### Main project areas

- `src/` — reusable data, feature, strategy, and modelling code.
- `scripts/` — experiment runners and research workflows.
- `tests/` — unit tests for core research components.
- `data/raw/` — locally stored market-data files.
- `artifacts/` — experiment outputs such as predictions, metrics, equity curves, portfolio results, and explainability files.
- `dashboard/` — Streamlit research interface.
- `notebooks/` — notebook-based exploratory research, including the earlier Stock ML Insights notebook when added.

Folder names may evolve as the project develops; the architecture above describes the workflow rather than requiring every optional module to be present.

## 3. Data Collection

The current research dataset contains daily OHLCV observations for:

| Symbol | Company |
|---|---|
| `RELIANCE.NS` | Reliance Industries |
| `HDFCBANK.NS` | HDFC Bank |
| `ICICIBANK.NS` | ICICI Bank |
| `INFY.NS` | Infosys |
| `TCS.NS` | Tata Consultancy Services |
| `SBIN.NS` | State Bank of India |

`NIFTYBEES.NS` is used as an ETF-based benchmark proxy in selected experiments. It should not be described as the official NIFTY index itself.

- Frequency: daily.
- Historical local dataset observed in the research runs: approximately 1,237 rows per symbol, from 2021-11-09 to 2026-11-09.
- Fields used include open, high, low, close, and volume, subject to availability in the source file.
- Market data is stored locally so experiments can be rerun without fetching data on every dashboard launch.

**Data note:** The exact source and refresh date should be checked against the data-ingestion implementation and the metadata accompanying each dataset before reproducing an experiment. Yahoo Finance/yfinance was used in the earlier standalone notebook; do not assume every current QuantEdge data file was collected through the same source unless the ingestion code confirms it.

## 4. Feature Engineering

The technical-indicator ML workflow creates a 32-feature set. Feature groups include:

| Feature group | Examples |
|---|---|
| Returns and momentum | 1-, 2-, 5-, 10-, 20-, and 60-session returns |
| Trend | SMA ratios and moving-average spreads |
| Momentum indicators | RSI(14), MACD, MACD signal, MACD histogram |
| Volatility | Bollinger Band width, ATR percentage, rolling volatility |
| Volume | Volume change, relative volume, rolling volume relationships |
| Benchmark-relative | NIFTYBEES returns and stock-relative returns, when available |

The technical-indicator ML experiment predicts whether the forward five-session return is positive:

`target = 1 if Close[t+5] / Close[t] - 1 > 0 else 0`

Feature preparation uses past and current information available at the prediction timestamp. Missing-value medians are learned from training data rather than the full dataset.

## 5. ML Model Architectures

QuantEdge compares models of different complexity. Models are included in a result table only when a corresponding saved experiment result exists.

### Tabular machine-learning models

- **Random Forest** — bagged decision trees used as a nonlinear baseline.
- **HistGradientBoosting** — gradient-boosted trees using histogram-based split finding.
- **XGBoost** — regularised gradient-boosted decision trees.
- **LightGBM** — leaf-wise gradient boosting for tabular data.

### Sequence and hybrid experiments

- **LSTM** — recurrent neural network architecture explored for sequential price-series modelling in the earlier Stock ML Insights notebook and in sequence-model research where corresponding outputs are available.
- **GRU** — gated recurrent architecture included in the sequence-model experiment set where results are available.
- **Hybrid ML engine** — pooled tree-based modelling across the selected stock universe, with ticker identity features and a rule-based RSI baseline for comparison.
- **Hybrid traditional strategies** — combinations of RSI, MACD, SMA, Bollinger Band signals, and a consensus approach.

The earlier **Stock ML Insights Report** notebook is part of the project's research history. It studied RELIANCE.NS using LSTM next-day price prediction, moving-average crossover, and technical indicators including RSI, MACD, and Bollinger Bands. Add that notebook under `notebooks/` as a clearly labelled earlier/standalone experiment; its results should not be mixed with current QuantEdge test results unless the date ranges and evaluation assumptions match.

## 6. Walk-Forward Validation

Time-series evaluation must respect chronology. Random train/test shuffling is not appropriate for these experiments because it can allow future observations to influence past predictions.

The technical-indicator ML experiment uses:

- 60% chronological training segment.
- 20% chronological validation segment.
- 20% final test segment.
- A five-observation purge at split boundaries to reduce overlap from the five-session forward target.
- Training-only median imputation.
- Model and probability-threshold selection on validation data.
- A frozen selected model/threshold for final test evaluation.

The pooled hybrid ML experiment also uses chronological date-based splits and a five-session purge. Model/threshold selection is performed using validation performance, not by repeatedly tuning against the final test period.

## 7. Market Regime Detection

Market-regime analysis is a useful extension for understanding whether strategies behave differently in trending, range-bound, or high-volatility conditions. However, a dedicated regime classifier, its input features, and its validated regime labels must be confirmed from the current source code and saved outputs before claiming it as an implemented QuantEdge capability.

For a reproducible regime experiment, document:

1. The regime definition (for example, trend and volatility states).
2. The features and lookback window used to assign a regime.
3. Whether regime thresholds are fitted using training data only.
4. Regime-wise sample counts and strategy performance.
5. How regime labels are made available at the time a trade is placed.

Do not use future returns to label a regime that is then treated as known at the same prediction timestamp.

## 8. Backtesting Methodology

The research backtests use explicit execution and cost assumptions rather than assuming every signal can trade at the same close used to calculate it.

### Traditional and hybrid traditional strategies

Strategies explored include:

- SMA 9/21, SMA 20/50, and SMA 50/200 crossovers.
- MACD.
- RSI(14) mean reversion.
- Bollinger Band mean reversion.
- Donchian breakout.
- RSI+MACD, SMA+RSI, Bollinger+RSI, and consensus combinations.

The strategy research uses long-only positions. Signals calculated at the close are applied from the next session. The documented assumptions are 0.10% one-way transaction cost plus 0.05% one-way slippage. Initial capital for the strategy pack is ₹10,000.

### Technical-indicator ML strategy

The ML strategy selects a probability threshold using validation Sharpe, then freezes it for the test period. Positions are based on predictions generated at the close and applied from the next session. The documented cost assumption is 0.10% one-way transaction cost plus 0.05% one-way slippage.

### Hybrid ML engine

The pooled hybrid engine compares its selected model against an RSI recovery baseline and equal-weight buy-and-hold over the same final test window. It uses a five-session holding period, skips overlapping trades per stock, and applies 0.15% cost per side.

These are simplified research assumptions. They do not fully model taxes, bid-ask spread dynamics, market impact, partial fills, liquidity constraints, or portfolio-level rebalancing costs.

## 9. Performance Metrics

QuantEdge reports predictive metrics and trading/portfolio metrics separately.

### Prediction metrics

- **ROC-AUC** — ranking quality of predicted probabilities.
- **Balanced accuracy** — average recall across the two target classes.
- **Probability threshold** — decision threshold selected on validation data.

### Trading and portfolio metrics

- **Total return** — cumulative strategy return over the evaluated period.
- **CAGR** — annualised compounded return.
- **Sharpe ratio** — excess return relative to return volatility, subject to the implementation's risk-free-rate convention.
- **Sortino ratio** — return relative to downside deviation.
- **Maximum drawdown** — largest peak-to-trough decline.
- **Final capital** — ending value from the stated initial capital.
- **Equity curve** — progression of portfolio value through the evaluation period.

A strong classification score does not automatically imply a profitable trading strategy. Likewise, a high historical return should be considered alongside drawdown, volatility, costs, benchmark performance, and sample length.

## 10. Experimental Comparison

The results below are taken from the recorded research runs. They are not live performance claims.

### Traditional strategy pack — equal-weight portfolio

| Strategy | Total return | CAGR | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| RSI(14) mean reversion | +34.09% | 6.04% | 0.97 | -11.18% |
| Buy and hold | +32.67% | 5.82% | 0.45 | -21.97% |
| SMA 50/200 | +17.06% | 3.20% | 0.41 | -11.26% |
| MACD | +14.10% | 2.67% | 0.34 | -16.16% |
| SMA 9/21 | +9.69% | 1.87% | 0.24 | -15.61% |
| Bollinger mean reversion | -1.95% | — | — | — |
| Donchian breakout | -3.34% | — | — | — |
| SMA 20/50 | -6.04% | — | — | — |

The strongest strategy in this full-period traditional experiment was RSI(14) mean reversion. The return was higher than the buy-and-hold result in this specific run, with a lower reported maximum drawdown. This is a historical result, not evidence that RSI will continue to outperform.

### Hybrid traditional strategy pack — equal-weight portfolio

| Strategy | Total return | Final capital (₹10,000 start) | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| SMA + RSI | +11.69% | ₹11,169 | 0.39 | -11.43% |
| Bollinger + RSI | -0.10% | ₹9,990 | 0.01 | -8.39% |
| RSI + MACD | -1.53% | ₹9,847 | -0.19 | -3.58% |
| Consensus | -6.25% | ₹9,375 | -0.32 | -12.52% |

In this run, combining indicators did not consistently improve on the standalone RSI strategy. The SMA+RSI combination was the strongest of the four hybrids.

### Technical-indicator ML — selected recorded test results

The technical-indicator ML run evaluated Random Forest, HistGradientBoosting, XGBoost, and LightGBM across the six-stock universe. Average ROC-AUC across the six per-stock runs was approximately:

| Model | Approx. mean ROC-AUC |
|---|---:|
| Random Forest | 0.522 |
| XGBoost | 0.509 |
| LightGBM | 0.496 |
| HistGradientBoosting | 0.485 |

The average AUC values indicate weak predictive separation in this experiment. Individual stock results varied, and several model/stock combinations had negative test-period trading returns. The experiment is retained as a useful negative/diagnostic result rather than presented as a successful predictive system.

### Hybrid ML engine — common final test window

| Metric | Hybrid ML | RSI baseline | Equal-weight buy and hold |
|---|---:|---:|---:|
| Total return | -8.47% | +2.35% | -14.56% |
| CAGR | -10.22% | +2.87% | -17.43% |
| Sharpe | -1.266 | +0.414 | -1.058 |
| Sortino | -1.790 | +0.395 | -1.752 |
| Maximum drawdown | -13.65% | -6.86% | -23.01% |
| Final capital (₹10,000 start) | ₹9,152.54 | ₹10,235.10 | ₹8,544.21 |

The hybrid ML engine recorded test ROC-AUC of **0.469** and balanced accuracy of **0.484**. HistGradientBoosting was selected on validation data with a probability threshold of 0.70. On this test window, the ML engine underperformed the RSI baseline. This is an important finding: added model complexity did not translate into better out-of-sample trading performance.

**Comparison caveat:** The traditional strategy figures cover the full historical research period, while the ML figures use shorter held-out test windows. These tables must not be interpreted as a direct head-to-head ranking across different time periods. A fair cross-family comparison requires rerunning every strategy over identical dates with identical cost assumptions.

## 11. Risk and Limitations

- Historical backtests are not predictions of future performance.
- Daily OHLCV data may contain adjustments, missing observations, or source inconsistencies.
- NIFTYBEES is an ETF proxy, not the official NIFTY index.
- The current transaction-cost and slippage model is simplified.
- Taxes, market impact, liquidity, bid-ask spread, and partial fills are not fully represented.
- Results can vary significantly by stock, period, and market condition.
- Some ML experiments show weak predictive performance and negative out-of-sample returns.
- Full-period traditional returns and shorter ML test-window returns are not directly comparable.
- A dedicated market-regime classifier should only be claimed after its implementation and validation outputs are included.
- Portfolio results depend on weighting and rebalancing assumptions; these must be read from the relevant experiment configuration.
- This repository is for research and learning, not personalised financial advice.

## 12. Technical Decisions

| Decision | Rationale |
|---|---|
| Python research stack | Mature ecosystem for data analysis, modelling, and backtesting |
| Pandas / NumPy | Time-series processing and feature calculations |
| Scikit-learn | Baseline models, preprocessing, and evaluation |
| XGBoost / LightGBM | Additional gradient-boosting comparisons where installed |
| TensorFlow / Keras | Sequence-model experiments such as LSTM in the notebook/research workflow |
| Chronological splits | Preserve time order and reduce look-ahead leakage |
| Validation-based threshold selection | Keep the final test set out of model/threshold selection |
| Signal execution on next session | Avoid assuming execution at the same close that generated the signal |
| Transaction costs and slippage | Make backtests less idealised |
| Saved artifacts | Keep dashboard loading separate from expensive model training |
| Streamlit | Present market views, experiment comparisons, portfolio analysis, and explainability |
| Unit tests | Check core feature/strategy behaviour and reduce regressions |

---

## Earlier Research: Stock ML Insights

The repository also includes the earlier **Stock ML Insights Report** notebook as a research-history/learning artifact. That standalone project focused on `RELIANCE.NS` and explored:

1. LSTM-based next-day stock-price prediction.
2. Moving-average crossover trading.
3. Technical-indicator analysis using RSI, MACD, and Bollinger Bands.
4. Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras, and yfinance.

Place the notebook in a path such as:

```text
notebooks/legacy_stock_ml_insights/stockMLinsightsreport.ipynb
```

If you also add its original visualisation images, keep them in a sibling `images/` folder and update the notebook's relative image paths if needed. This earlier notebook is supplementary research; its single-stock results should remain distinct from the multi-stock QuantEdge experiments.

## Repository Layout

```text
QuantEdge/
├── dashboard/
├── data/
│   └── raw/
├── notebooks/
│   └── legacy_stock_ml_insights/
├── scripts/
├── src/
├── tests/
├── artifacts/
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<YOUR_GITHUB_USERNAME>/QuantEdge.git
cd QuantEdge
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If the repository does not yet contain a complete `requirements.txt`, create one from the working environment before publishing:

```bash
pip freeze > requirements.txt
```

Review that file and remove machine-specific or unnecessary packages before committing.

### 4. Run the dashboard

From the repository root:

Windows PowerShell:

```powershell
$env:PYTHONPATH = "src"
streamlit run dashboard/app.py
```

The dashboard reads saved local data and experiment artifacts. Some pages will show only the experiments for which the corresponding result files are present.

### 5. Run tests

```bash
python -m pytest -q
```

### Reproducing experiments

Use the relevant experiment scripts in `scripts/`. Run them only after installing the required dependencies and confirming that the input market data is available. Each experiment's README or script arguments should be treated as the source of truth for its exact configuration.

---

## Author

**Tarun Yadav**  
B.Tech, Computer Science & Engineering — Delhi Technological University

For questions, suggestions, or collaboration, feel free to open an issue or connect through GitHub.
