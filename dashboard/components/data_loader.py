from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

# Expected location: <project-root>/dashboard/components/data_loader.py
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw"
ARTIFACT_DIR = ROOT / "artifacts"
TICKERS = ["RELIANCE.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS", "TCS.NS", "SBIN.NS"]
STOCKS = [t.replace(".NS", "") for t in TICKERS]

def _read_csv(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.empty: return df
        date_col = next((c for c in df.columns if str(c).lower() in ("date","datetime","timestamp","time")), None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            if date_col != "Date": df = df.rename(columns={date_col:"Date"})
        return df
    except Exception:
        return pd.DataFrame()

def load_table(relative_path: str) -> pd.DataFrame:
    p = ROOT / relative_path
    return _read_csv(p) if p.exists() and p.suffix.lower() == ".csv" else pd.DataFrame()

def load_market_data(ticker: str) -> pd.DataFrame:
    stem = ticker.replace(".", "_").replace("-", "_")
    for p in (DATA_DIR/f"{stem}_1d.csv", DATA_DIR/f"{stem}.csv", DATA_DIR/f"{ticker}_1d.csv"):
        df = _read_csv(p)
        if df.empty: continue
        rename = {}
        for c in df.columns:
            low = str(c).lower().replace("_"," ")
            if low in ("open","high","low","close","volume"): rename[c] = low.title()
        df = df.rename(columns=rename)
        if {"Date","Open","High","Low","Close"}.issubset(df.columns):
            if "Volume" not in df: df["Volume"] = 0
            return df.dropna(subset=["Date","Close"]).sort_values("Date")
    return pd.DataFrame()

def _artifact(relative: str) -> pd.DataFrame:
    return load_table(f"artifacts/{relative}")

def traditional_metrics():
    return _artifact("traditional_strategy_research/equal_weight_portfolio_metrics.csv")
def traditional_stock_metrics():
    return _artifact("traditional_strategy_research/per_stock_strategy_metrics.csv")
def traditional_equity():
    return _artifact("traditional_strategy_research/equal_weight_portfolio_equity_curves.csv")

def hybrid_traditional_metrics():
    return _artifact("hybrid_traditional_research/equal_weight_portfolio_metrics.csv")
def hybrid_traditional_stock_metrics():
    return _artifact("hybrid_traditional_research/per_stock_strategy_metrics.csv")

def technical_ml_metrics():
    return _artifact("technical_indicator_ml/model_test_metrics.csv")
def technical_ml_strategy_metrics():
    return _artifact("technical_indicator_ml/strategy_test_metrics.csv")
def technical_ml_portfolio_metrics():
    return _artifact("technical_indicator_ml/equal_weight_portfolio_test_metrics.csv")

def hybrid_ml_metrics():
    return _artifact("hybrid_ml_engine/test_strategy_metrics.csv")

def sequence_metrics():
    return _artifact("phase11_sequence_models/reports/phase11_model_summary.csv")
def ensemble_metrics():
    return _artifact("phase12_hybrid_ensemble/reports/phase12_pooled_metrics.csv")
def ensemble_stock_metrics():
    return _artifact("phase12_hybrid_ensemble/reports/phase12_stock_model_metrics.csv")

def shap_importance():
    p = ARTIFACT_DIR / "phase7_explainability" / "RELIANCE_NS"
    frames = []
    if p.exists():
        for f in p.glob("*/shap_importance.csv"):
            d = _read_csv(f)
            if not d.empty:
                d.insert(0, "Model", f.parent.name)
                frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def portfolio_data():
    p = ARTIFACT_DIR / "phase6_portfolio" / "RELIANCE_NS"
    weights = _read_csv(p/"portfolio_weights.csv")
    performance = _read_csv(p/"portfolio_performance.csv")
    returns = _read_csv(p/"portfolio_daily_returns.csv")
    return weights, performance, returns

def project_snapshot():
    market_count = len(list(DATA_DIR.glob("*.csv"))) if DATA_DIR.exists() else 0
    artifact_count = sum(1 for p in ARTIFACT_DIR.rglob("*") if p.is_file()) if ARTIFACT_DIR.exists() else 0
    return {"market_files":market_count, "artifact_files":artifact_count, "stocks":len(TICKERS)}
