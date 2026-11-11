import pandas as pd
import pytest

from quantedge.data.validator import (
    MarketDataValidator
)


@pytest.fixture
def valid_data():

    dates = pd.date_range(
        start="2025-01-01",
        periods=5
    )

    return pd.DataFrame(
        {
            "Open": [100, 102, 103, 105, 106],
            "High": [105, 106, 108, 109, 110],
            "Low": [98, 100, 101, 103, 104],
            "Close": [103, 104, 106, 107, 109],
            "Volume": [1000, 1200, 1100, 1300, 1400]
        },
        index=dates
    )


def test_valid_data(valid_data):

    validator = MarketDataValidator()

    assert validator.validate(valid_data) is True


def test_empty_data():

    validator = MarketDataValidator()

    with pytest.raises(ValueError):

        validator.validate(pd.DataFrame())


def test_negative_price(valid_data):

    validator = MarketDataValidator()

    valid_data.loc[
        valid_data.index[0],
        "Close"
    ] = -10

    with pytest.raises(ValueError):

        validator.validate(valid_data)


def test_missing_values(valid_data):

    validator = MarketDataValidator()

    valid_data.loc[
        valid_data.index[0],
        "Close"
    ] = None

    with pytest.raises(ValueError):

        validator.validate(valid_data)