
import streamlit as st
import pandas as pd
import plotly.express as px

from components.data_loader import (
    load_phase6_returns
)


st.title("📈 Strategy Backtesting")

returns = load_phase6_returns()

if returns is None:

    st.warning(
        "Phase 6 daily portfolio returns not found."
    )

    st.stop()


st.subheader("Portfolio Returns")

st.dataframe(
    returns.tail(50),
    use_container_width=True
)

equity = (
    1 + returns.fillna(0)
).cumprod()

st.subheader("Equity Curves")

fig = px.line(
    equity,
    title="Historical Portfolio Growth"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.subheader("Drawdown Analysis")

drawdown = (
    equity / equity.cummax()
) - 1

fig = px.line(
    drawdown,
    title="Portfolio Drawdown"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.download_button(
    "Download Backtest Returns",
    data=returns.to_csv().encode("utf-8"),
    file_name="quantedge_backtest_returns.csv",
    mime="text/csv"
)
