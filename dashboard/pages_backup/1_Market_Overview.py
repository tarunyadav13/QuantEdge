
import streamlit as st
import pandas as pd

from components.data_loader import (
    load_market_data,
    TICKERS
)

from components.charts import candlestick_chart


st.title("📊 Market Overview")

ticker = st.session_state.get(
    "selected_ticker",
    TICKERS[0]
)

df = load_market_data(ticker)

df = df.copy()

df.index = pd.to_datetime(df.index)

period = st.selectbox(
    "Historical Period",
    ["3 Months", "6 Months", "1 Year", "3 Years", "5 Years"]
)

days = {
    "3 Months": 90,
    "6 Months": 180,
    "1 Year": 365,
    "3 Years": 1095,
    "5 Years": 1825
}

df = df.tail(
    days[period]
)

st.subheader(
    f"{ticker} Historical Price"
)

st.plotly_chart(
    candlestick_chart(df, ticker),
    use_container_width=True
)

st.subheader("Historical Data")

st.dataframe(
    df.tail(30),
    use_container_width=True
)

if "Volume" in df.columns:

    st.subheader("Trading Volume")

    st.bar_chart(
        df["Volume"]
    )

st.download_button(
    "Download Historical Data",
    data=df.to_csv().encode("utf-8"),
    file_name=f"{ticker}_historical_data.csv",
    mime="text/csv"
)
