
import streamlit as st
import json

from pathlib import Path

from components.data_loader import (
    load_shap_importance,
    load_shap_explanation,
    TICKERS
)

from components.charts import feature_importance_chart


st.title("🧠 Explainable AI")

ticker = st.session_state.get(
    "selected_ticker",
    TICKERS[0]
)

model = st.selectbox(
    "Select Model",
    [
        "RandomForest",
        "XGBoost",
        "LightGBM"
    ]
)

importance = load_shap_importance(
    ticker,
    model
)

explanation = load_shap_explanation(
    ticker,
    model
)

if importance is None:

    st.warning(
        "SHAP results not found for this stock."
    )

    st.stop()


st.subheader(
    f"{model} Feature Importance"
)

st.plotly_chart(
    feature_importance_chart(importance),
    use_container_width=True
)

st.subheader("Top 10 Features")

st.dataframe(
    importance.head(10),
    use_container_width=True
)


st.divider()

st.subheader("Individual Prediction Explanation")

if explanation is not None:

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ticker",
        explanation["Ticker"]
    )

    col2.metric(
        "Probability Up",
        f"{explanation['Probability Up']:.2%}"
    )

    col3.metric(
        "Prediction",
        explanation["Prediction"]
    )

    st.markdown("### Positive Contributors")

    st.dataframe(
        explanation["Positive Contributors"],
        use_container_width=True
    )

    st.markdown("### Negative Contributors")

    st.dataframe(
        explanation["Negative Contributors"],
        use_container_width=True
    )

else:

    st.warning(
        "Individual prediction explanation not found."
    )


st.divider()

st.subheader("SHAP Visualizations")

ticker_folder = ticker.replace(".", "_")

plot_dir = (
    Path("artifacts") /
    "phase7_explainability" /
    ticker_folder /
    model
)

plots = [
    ("SHAP Summary", "shap_summary.png"),
    ("Feature Importance", "shap_bar.png"),
    ("Individual Prediction", "shap_waterfall_0.png")
]

for title, filename in plots:

    path = plot_dir / filename

    if path.exists():

        st.markdown(f"#### {title}")

        st.image(
            str(path),
            use_container_width=True
        )

    else:

        st.warning(
            f"{filename} not found."
        )
