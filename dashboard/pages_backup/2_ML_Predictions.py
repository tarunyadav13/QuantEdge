
import streamlit as st
import pandas as pd
import plotly.express as px

from components.data_loader import (
    load_phase5_predictions,
    TICKERS
)


st.title("🤖 Machine Learning Predictions")

ticker = st.session_state.get(
    "selected_ticker",
    TICKERS[0]
)

predictions = load_phase5_predictions(
    ticker
)

if not predictions:

    st.warning(
        "No Phase 5 walk-forward predictions found "
        "for this ticker."
    )

    st.stop()


model = st.selectbox(
    "Select Model",
    list(predictions.keys())
)

df = predictions[model].copy()

st.subheader(
    f"{model} — Walk-Forward Predictions"
)

if "probability_up" not in df.columns:

    st.error(
        "Probability column missing."
    )

    st.stop()


latest = df.iloc[-1]

probability = float(
    latest["probability_up"]
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Probability Up",
    f"{probability:.2%}"
)

col2.metric(
    "Latest Actual Direction",
    str(latest["actual"])
)

col3.metric(
    "Prediction",
    "BULLISH" if probability >= 0.5 else "BEARISH"
)

st.info(
    "This page displays historical saved predictions. "
    "The latest row is not a newly generated live prediction."
)

fig = px.line(
    df,
    x="date",
    y="probability_up",
    title="Historical Predicted Probability"
)

fig.add_hline(
    y=0.5,
    line_dash="dash"
)

st.plotly_chart(
    fig,
    use_container_width=True
)

st.subheader("Prediction History")

st.dataframe(
    df,
    use_container_width=True
)

st.download_button(
    "Download Predictions",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name=f"{model}_predictions.csv",
    mime="text/csv"
)
