from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, brier_score_loss
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantedge.ml.technical_ml import (  # noqa: E402
    build_technical_features, chronological_purged_split, normalize_market_data,
)

TICKERS = ["RELIANCE", "HDFCBANK", "ICICIBANK", "INFY", "TCS", "SBIN"]
HORIZON = 5
INITIAL_CAPITAL = 10_000.0
TRANSACTION_COST = 0.001
SLIPPAGE = 0.0005
THRESHOLDS = [0.50, 0.52, 0.55, 0.57, 0.60, 0.62, 0.65, 0.68, 0.70]


def locate_csv(ticker):
    for name in (f"{ticker}_NS_1d.csv", f"{ticker}.NS_1d.csv",
                 f"{ticker}_NS.csv", f"{ticker}.NS.csv"):
        p = ROOT / "data/raw" / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not find raw CSV for {ticker} in data/raw.")


def load_benchmark():
    for name in ("NIFTYBEES_NS_1d.csv", "NIFTYBEES.NS_1d.csv",
                 "NIFTYBEES_NS.csv", "NIFTYBEES.NS.csv"):
        p = ROOT / "data/raw" / name
        if p.exists():
            d = normalize_market_data(pd.read_csv(p))
            return d["Close"]
    return None


def make_models():
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=6, min_samples_leaf=10,
            max_features="sqrt", class_weight="balanced_subsample",
            random_state=42, n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=180, learning_rate=0.04, max_leaf_nodes=15,
            l2_regularization=2.0, random_state=42,
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.025,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=8,
            reg_lambda=5.0, objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
    except Exception as exc:
        print(f"XGBoost skipped (not installed/importable): {exc}")
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(
            n_estimators=350, learning_rate=0.025, num_leaves=15,
            max_depth=4, min_child_samples=20, reg_lambda=5.0,
            verbosity=-1, random_state=42, n_jobs=-1,
        )
    except Exception as exc:
        print(f"LightGBM skipped (not installed/importable): {exc}")
    return models


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan


def strategy_backtest(close, probabilities, threshold):
    # Prediction made after close[t], held from t+1.
    p = probabilities.reindex(close.index)
    target_position = (p >= threshold).astype(int)
    held = target_position.shift(1).fillna(0)
    asset_ret = close.pct_change().fillna(0)
    turnover = held.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(held.iloc[0])
    cost = TRANSACTION_COST + SLIPPAGE
    net = held * asset_ret - turnover.fillna(0) * cost
    equity = INITIAL_CAPITAL * (1 + net).cumprod()
    dd = equity / equity.cummax() - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1/252)
    std = net.std(ddof=1)
    return {
        "total_return": float(equity.iloc[-1] / INITIAL_CAPITAL - 1),
        "cagr": float((equity.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1),
        "final_equity": float(equity.iloc[-1]),
        "sharpe_ratio": float(net.mean() / std * np.sqrt(252)) if std > 0 else 0.0,
        "max_drawdown": float(dd.min()),
        "average_exposure": float(held.mean()),
        "position_changes": int((turnover.fillna(0) > 0).sum()),
        "estimated_cost_paid": float((turnover.fillna(0) * cost).sum()),
        "observations": int(len(close)),
    }, pd.DataFrame({
        "close": close, "probability_up_5d": p, "target_position": target_position,
        "held_position": held, "daily_asset_return": asset_ret,
        "net_strategy_return": net, "equity": equity, "drawdown": dd,
    })


def choose_threshold(model, X_val, y_val, close_val):
    p = pd.Series(model.predict_proba(X_val)[:, 1], index=X_val.index)
    scored = []
    for threshold in THRESHOLDS:
        metrics, _ = strategy_backtest(close_val, p, threshold)
        # Select on validation Sharpe, with return as tie-breaker.
        scored.append((threshold, metrics["sharpe_ratio"], metrics["total_return"]))
    return sorted(scored, key=lambda row: (row[1], row[2]), reverse=True)[0][0]


def main():
    out = ROOT / "artifacts/technical_indicator_ml"
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    (out / "equity_curves").mkdir(parents=True, exist_ok=True)
    benchmark = load_benchmark()
    model_rows, strategy_rows = [], []
    model_names_seen = set()

    for ticker in TICKERS:
        raw = pd.read_csv(locate_csv(ticker))
        normalized = normalize_market_data(raw)
        frame = build_technical_features(normalized, benchmark_close=benchmark, horizon=HORIZON)
        feature_cols = [c for c in frame.columns if c not in {"target", "forward_return"}]
        # Remove rows where indicators/label are unavailable; all feature construction is causal.
        frame = frame.dropna(subset=feature_cols + ["target", "forward_return"]).copy()
        train, val, test = chronological_purged_split(frame, purge=HORIZON)
        X_train, y_train = train[feature_cols], train["target"].astype(int)
        X_val, y_val = val[feature_cols], val["target"].astype(int)
        X_test, y_test = test[feature_cols], test["target"].astype(int)
        if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
            print(f"{ticker}: skipped because one chronological split has only one target class.")
            continue

        print(f"\n{ticker}: samples train={len(train)}, validation={len(val)}, test={len(test)}, features={len(feature_cols)}")
        predictions = pd.DataFrame(index=test.index)
        predictions["actual_target_up_5d"] = y_test
        predictions["forward_return_5d"] = test["forward_return"]

        for model_name, estimator in make_models().items():
            # Imputation is fitted on training data only. No scaling is required for tree models.
            model = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("model", estimator),
            ])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, y_train)
            threshold = choose_threshold(model, X_val, y_val, val["Close"] if "Close" in val else normalized["Close"].reindex(val.index))
            # Test predictions are generated once; threshold is frozen from validation.
            p_test = pd.Series(model.predict_proba(X_test)[:, 1], index=test.index)
            predictions[f"{model_name}_probability"] = p_test
            metrics = {
                "ticker": ticker, "model": model_name, "horizon_days": HORIZON,
                "threshold_selected_on_validation": threshold,
                "accuracy": accuracy_score(y_test, (p_test >= 0.5).astype(int)),
                "balanced_accuracy": balanced_accuracy_score(y_test, (p_test >= 0.5).astype(int)),
                "roc_auc": safe_auc(y_test, p_test),
                "log_loss": log_loss(y_test, np.clip(p_test, 1e-6, 1-1e-6), labels=[0, 1]),
                "brier_score": brier_score_loss(y_test, p_test),
                "train_samples": len(train), "validation_samples": len(val), "test_samples": len(test),
                "feature_count": len(feature_cols),
            }
            model_rows.append(metrics)
            model_names_seen.add(model_name)

            close_test = normalized["Close"].reindex(test.index)
            strat_metrics, curve = strategy_backtest(close_test, p_test, threshold)
            strategy_rows.append({"ticker": ticker, "model": model_name, **strat_metrics,
                                  "validation_selected_threshold": threshold})
            curve.to_csv(out / "equity_curves" / f"{ticker}_{model_name}_test.csv", index_label="date")
            print(f"  {model_name:<20} AUC={metrics['roc_auc']:.3f} "
                  f"BalAcc={metrics['balanced_accuracy']:.3f} "
                  f"test return={strat_metrics['total_return']:+.2%} "
                  f"threshold={threshold:.2f}")

        predictions.to_csv(out / "predictions" / f"{ticker}_test_predictions.csv", index_label="date")

    model_df = pd.DataFrame(model_rows)
    strategy_df = pd.DataFrame(strategy_rows)
    model_df.to_csv(out / "model_test_metrics.csv", index=False)
    strategy_df.to_csv(out / "strategy_test_metrics.csv", index=False)

    if not strategy_df.empty:
        portfolio_rows = []
        for model_name in sorted(strategy_df["model"].unique()):
            curves = []
            for ticker in TICKERS:
                p = out / "equity_curves" / f"{ticker}_{model_name}_test.csv"
                if p.exists():
                    c = pd.read_csv(p, parse_dates=["date"]).set_index("date")
                    curves.append(c["net_strategy_return"].rename(ticker))
            if len(curves) < 2:
                continue
            daily = pd.concat(curves, axis=1).dropna()
            port_ret = daily.mean(axis=1)
            equity = INITIAL_CAPITAL * (1 + port_ret).cumprod()
            dd = equity / equity.cummax() - 1
            years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1/252)
            std = port_ret.std(ddof=1)
            portfolio_rows.append({
                "model": model_name, "total_return": equity.iloc[-1] / INITIAL_CAPITAL - 1,
                "cagr": (equity.iloc[-1] / INITIAL_CAPITAL) ** (1/years) - 1,
                "final_equity": equity.iloc[-1],
                "sharpe_ratio": port_ret.mean() / std * np.sqrt(252) if std > 0 else 0,
                "max_drawdown": dd.min(), "observations": len(port_ret),
                "number_of_stocks": len(curves),
            })
            pd.DataFrame({"daily_return": port_ret, "equity": equity, "drawdown": dd}).to_csv(
                out / "equity_curves" / f"PORTFOLIO_{model_name}_test.csv", index_label="date"
            )
        pd.DataFrame(portfolio_rows).sort_values("total_return", ascending=False).to_csv(
            out / "equal_weight_portfolio_test_metrics.csv", index=False
        )

    assumptions = {
        "prediction_target": f"Whether close[t+{HORIZON}] / close[t] - 1 is positive",
        "feature_timing": "Indicators use data available through close[t]",
        "execution": "Probability computed after close[t]; position applied from t+1",
        "split": "Chronological 60% train / 20% validation / 20% test with 5-observation purge gaps",
        "threshold_selection": "Threshold chosen on validation Sharpe only; frozen for test",
        "preprocessing": "Median imputer fitted separately inside each model pipeline on training rows only",
        "costs_one_way": {"transaction_cost": TRANSACTION_COST, "slippage": SLIPPAGE},
        "models": sorted(model_names_seen),
        "features": "Causal technical indicators, returns, volatility, volume and optional NIFTYBEES benchmark context",
        "limitations": [
            "Test period is a single chronological holdout, not proof of future profitability.",
            "Daily strategy positions use 5-day direction probabilities; overlapping labels are purged at split boundaries.",
            "Equal-weight portfolio is an average of stock daily net returns; explicit periodic rebalancing costs are not modelled.",
            "HistGradientBoosting is included as a scikit-learn gradient-boosting fallback/comparator.",
            "No hyperparameter optimization is performed; parameters are deliberately conservative starting points."
        ],
    }
    (out / "assumptions.json").write_text(json.dumps(assumptions, indent=2), encoding="utf-8")
    print(f"\nCompleted. Reports saved to: {out}")
    print("Primary files: model_test_metrics.csv, strategy_test_metrics.csv, equal_weight_portfolio_test_metrics.csv")


if __name__ == "__main__":
    main()
