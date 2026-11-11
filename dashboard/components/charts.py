"""Reusable Plotly charts for QuantEdge."""
import plotly.graph_objects as go
import plotly.express as px

DARK = "plotly_dark"

def candlestick_chart(df, title="Price history"):
    fig = go.Figure(go.Candlestick(x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="OHLC"))
    for n in (20, 50):
        if len(df) >= n:
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"].rolling(n).mean(),
                                     mode="lines", name=f"SMA {n}"))
    fig.update_layout(title=title, template=DARK, height=500, xaxis_rangeslider_visible=False)
    return fig

def comparison_bar(df, category, value, title, percent=False):
    fig = px.bar(df, x=category, y=value, color=category, title=title, text_auto=".2% " if percent else ".3f")
    fig.update_layout(template=DARK, height=420, showlegend=False, xaxis_tickangle=-25)
    return fig

def equity_curve_chart(df, x="Date", y="equity", title="Equity curve"):
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, template=DARK, height=420)
    return fig
