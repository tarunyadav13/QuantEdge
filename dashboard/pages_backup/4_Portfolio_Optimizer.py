
import streamlit as st
import pandas as pd
import plotly.express as px

from components.data_loader import (
    load_phase6_weights,
    load_phase6_metrics
)


st.title("💼 Portfolio Optimization")

weights = load_phase6_weights()

metrics = load_phase6_metrics()

if weights is None:

    st.warning(
        "Portfolio allocation results not found. "
        "Run Phase 6 first."
    )

    st.stop()


st.subheader("Portfolio Allocations")

st.dataframe(
    weights.style.format("{:.2%}"),
    use_container_width=True
)

fig = px.bar(
    weights.reset_index(),
    x=weights.index.name or "index",
    y=weights.columns.tolist(),
    barmode="group",
    title="Portfolio Allocation Comparison"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


st.divider()

st.subheader("Portfolio Performance")

if metrics is not None:

    st.dataframe(
        metrics,
        use_container_width=True
    )

    selected = st.selectbox(
        "Choose Portfolio",
        metrics.index.tolist()
    )

    selected_metrics = metrics.loc[selected]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Return",
        f"{selected_metrics['Total Return']:.2%}"
    )

    col2.metric(
        "Sharpe Ratio",
        f"{selected_metrics['Sharpe Ratio']:.3f}"
    )

    col3.metric(
        "Maximum Drawdown",
        f"{selected_metrics['Maximum Drawdown']:.2%}"
    )

else:

    st.warning(
        "Portfolio performance metrics unavailable."
    )

st.info(
    "These are historical Phase 6 portfolio results, "
    "not current recommended allocations."
)
