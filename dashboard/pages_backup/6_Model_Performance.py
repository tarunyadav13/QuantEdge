
import streamlit as st
import pandas as pd
import plotly.express as px

from pathlib import Path

from components.data_loader import TICKERS


st.title("📉 Model Performance")

ticker = st.session_state.get(
    "selected_ticker",
    TICKERS[0]
)

ticker_folder = ticker.replace(".", "_")

base = (
    Path("artifacts") /
    "phase5_complete_validation" /
    ticker_folder
)

files = {

    "Calibration Metrics": "calibration_metrics.csv",

    "Strategy Metrics": "nested_strategy_metrics.csv",

    "Cost Sensitivity": "cost_sensitivity.csv"

}

for title, filename in files.items():

    path = base / filename

    st.subheader(title)

    if not path.exists():

        st.warning(
            f"{filename} not found."
        )

        continue

    df = pd.read_csv(path)

    st.dataframe(
        df,
        use_container_width=True
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    if numeric_columns:

        selected_metric = st.selectbox(
            f"Visualize {title}",
            numeric_columns,
            key=title
        )

        fig = px.bar(
            df,
            y=selected_metric,
            title=f"{selected_metric} — {title}"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )
