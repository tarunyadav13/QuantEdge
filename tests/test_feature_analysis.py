import numpy as np
import pandas as pd

from quantedge.features import (
    CorrelationAnalyzer,
    FeatureSelector,
    FeatureImportanceAnalyzer
)


def generate_data():

    np.random.seed(42)

    dates = pd.date_range(
        "2023-01-01",
        periods=300
    )

    base = np.random.normal(
        0,
        0.01,
        300
    )

    stock_a = 100 * np.exp(
        np.cumsum(base)
    )

    stock_b = 100 * np.exp(
        np.cumsum(
            base + np.random.normal(0, 0.003, 300)
        )
    )

    stock_c = 100 * np.exp(
        np.cumsum(
            np.random.normal(0, 0.015, 300)
        )
    )

    prices = pd.DataFrame({
        "STOCK_A": stock_a,
        "STOCK_B": stock_b,
        "STOCK_C": stock_c
    }, index=dates)

    return prices


def test_correlation():

    prices = generate_data()

    analyzer = CorrelationAnalyzer()

    results = analyzer.analyze(prices)

    assert results["pearson"].shape == (3, 3)

    assert results["spearman"].shape == (3, 3)

    assert not results["highly_correlated_pairs"].empty


def test_feature_selection():

    np.random.seed(42)

    X = pd.DataFrame(
        np.random.randn(300, 10),
        columns=[f"feature_{i}" for i in range(10)]
    )

    y = (X["feature_0"] > 0).astype(int)

    selector = FeatureSelector()

    selected = selector.fit_transform(X, y)

    assert selected.shape[1] > 0

    assert len(selector.feature_scores) > 0


def test_feature_importance():

    np.random.seed(42)

    X = pd.DataFrame(
        np.random.randn(300, 5),
        columns=[f"feature_{i}" for i in range(5)]
    )

    y = (X["feature_0"] > 0).astype(int)

    analyzer = FeatureImportanceAnalyzer()

    result = analyzer.analyze(
        X.iloc[:200],
        y.iloc[:200],
        X.iloc[200:],
        y.iloc[200:]
    )

    assert result["importance"].shape[0] == 5

    assert 0 <= result["accuracy"] <= 1