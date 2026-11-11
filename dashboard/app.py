from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from components.data_loader import (
    TICKERS, load_market_data, project_snapshot, traditional_metrics,
    traditional_stock_metrics, traditional_equity, hybrid_traditional_metrics,
    hybrid_traditional_stock_metrics, technical_ml_metrics,
    technical_ml_strategy_metrics, technical_ml_portfolio_metrics,
    hybrid_ml_metrics, sequence_metrics, ensemble_metrics,
    ensemble_stock_metrics, shap_importance, portfolio_data
)

st.set_page_config(page_title="QuantEdge | Research Terminal", page_icon="◆", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
:root{--bg:#080e19;--panel:#111c2d;--line:#263c57;--muted:#9aabc1;--teal:#42dfcf}
html,body,[class*="css"]{font-family:Manrope,sans-serif}
.stApp{background:radial-gradient(circle at 15% 0%,#142943 0%,#080e19 45%);color:#e8effb}
.block-container{max-width:1500px;padding-top:1.6rem}
[data-testid="stSidebar"]{background:#0b1422;border-right:1px solid #20314a}
h1,h2,h3{letter-spacing:-.035em}
.hero{background:linear-gradient(120deg,#1b3551,#0f1c2e);border:1px solid #2b4665;border-radius:20px;padding:28px 32px;margin-bottom:20px}
.eyebrow{color:var(--teal);font: .72rem 'DM Mono';letter-spacing:.14em;text-transform:uppercase}
.card{background:linear-gradient(145deg,#15253a,#101b2b);border:1px solid #263c57;border-radius:14px;padding:16px;min-height:104px}
.label{color:#91a5bf;font: .68rem 'DM Mono';text-transform:uppercase;letter-spacing:.1em}
.value{color:#f0f5ff;font:1.5rem 'DM Mono';margin-top:9px}
.note{color:#8ea5bd;font-size:.76rem;margin-top:4px}
.section{margin:28px 0 12px;padding-bottom:8px;border-bottom:1px solid #263c57}
.info{background:#101e30;border:1px solid #2a4562;border-left:3px solid var(--teal);padding:12px 15px;border-radius:9px;color:#b8c7d9}
</style>
""", unsafe_allow_html=True)

def pct(x):
    try: return f"{float(x):+.2%}"
    except: return "—"
def num(x, digits=3):
    try: return f"{float(x):,.{digits}f}"
    except: return "—"
def rupee(x):
    try: return f"₹{float(x):,.0f}"
    except: return "—"
def metric(label, value, note=""):
    st.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="note">{note}</div></div>', unsafe_allow_html=True)
def heading(kicker,title,subtitle):
    st.markdown(f'<div class="eyebrow">{kicker}</div><h1 style="margin:.2rem 0">{title}</h1><p style="color:#9fb0c6">{subtitle}</p>',unsafe_allow_html=True)
def section(title):
    st.markdown(f'<div class="section"><h3>{title}</h3></div>',unsafe_allow_html=True)
def show_table(df, cols=None):
    if df is None or df.empty:
        st.info("No saved result is available for this experiment in artifacts/.")
        return
    d=df.copy()
    if cols: d=d[[c for c in cols if c in d.columns]]
    st.dataframe(d, use_container_width=True, hide_index=True)
def fmt_metrics(df, return_col="total_return"):
    if df.empty:return df
    d=df.copy()
    for c in d.columns:
        if c.lower() in ("total_return","cagr","max_drawdown","annualized_volatility","daily_win_rate"):
            d[c]=d[c].map(pct)
        elif c.lower() in ("final_equity","buy_hold_final_equity"):
            d[c]=d[c].map(rupee)
        elif c.lower() in ("sharpe_ratio","sortino_ratio","sharpe","sortino","roc_auc","balanced_accuracy","accuracy","precision","recall","f1","log_loss","brier_score"):
            d[c]=d[c].map(lambda x:num(x,3))
    return d

def overview():
    snap=project_snapshot()
    tm=traditional_metrics(); mm=technical_ml_portfolio_metrics(); hm=hybrid_ml_metrics()
    heading("QuantEdge · Quantitative research terminal","Research overview","A results-first view of systematic strategies, ML experiments and portfolio research across Indian equities.")
    st.markdown('<div class="hero"><div class="eyebrow">NSE EQUITY RESEARCH · EXPERIMENT-DRIVEN</div><h2>From technical rules to machine-learning strategies.</h2><p>Six-stock research universe covering traditional indicators, hybrid signals, tree-based ML, sequence models, ensemble experiments, realistic backtesting, portfolio analysis and explainability. Every displayed result is loaded from saved experiment outputs.</p></div>',unsafe_allow_html=True)
    c=st.columns(4)
    with c[0]:metric("Stocks covered",str(len(TICKERS)),"RELIANCE · HDFCBANK · ICICIBANK · INFY · TCS · SBIN")
    with c[1]:metric("Market data files",str(snap["market_files"]),"Local daily OHLCV")
    with c[2]:metric("Saved experiment files",str(snap["artifact_files"]),"From artifacts/")
    with c[3]:metric("Research streams","7","Rules · Hybrid · ML · Sequence · Ensemble · Portfolio · XAI")
    section("Selected research results")
    a,b,c=st.columns(3)
    if not tm.empty:
        best=tm.sort_values("sharpe_ratio",ascending=False).iloc[0]
        with a:metric("Top traditional strategy",str(best["strategy"]),f"Return {pct(best['total_return'])} · Sharpe {num(best['sharpe_ratio'])}")
    if not mm.empty:
        best=mm.sort_values("sharpe_ratio",ascending=False).iloc[0]
        with b:metric("Best technical ML portfolio",str(best["model"]),f"Test return {pct(best['total_return'])} · Sharpe {num(best['sharpe_ratio'])}")
    if not hm.empty:
        row=hm[hm["strategy"].str.contains("Hybrid ML",case=False)]
        if not row.empty:
            r=row.iloc[0]
            with c:metric("Hybrid ML test result",pct(r["total_return"]),f"AUC {num(r.get('test_auc'))} · {r.get('model','')}")
    section("Research workflow")
    st.markdown("**Market data → Feature engineering → Traditional baselines → Hybrid rules → ML & sequence models → Ensemble evaluation → Trading simulation → Portfolio risk → Explainability**")
    st.markdown('<div class="info">Performance figures are historical experiment outputs, not forecasts. Full-period rule-based results and shorter ML test-window results are not directly comparable unless dates and cost assumptions match.</div>',unsafe_allow_html=True)

def market_page():
    heading("Market explorer","Market overview","Inspect local daily OHLCV data for the six-stock NSE universe.")
    ticker=st.selectbox("Equity",TICKERS)
    df=load_market_data(ticker)
    if df.empty: st.error(f"Could not load {ticker}. Check data/raw/<TICKER>_1d.csv.");return
    r=df.iloc[-1]; cols=st.columns(4)
    with cols[0]:metric("Latest close",rupee(r["Close"]),str(pd.to_datetime(r["Date"]).date()))
    with cols[1]:metric("Session high",rupee(r["High"]))
    with cols[2]:metric("Session low",rupee(r["Low"]))
    with cols[3]:metric("Volume",f"{float(r.get('Volume',0)):,.0f}")
    fig=go.Figure(go.Candlestick(x=df.Date,open=df.Open,high=df.High,low=df.Low,close=df.Close,name=ticker))
    for n in (20,50):fig.add_trace(go.Scatter(x=df.Date,y=df.Close.rolling(n).mean(),name=f"SMA {n}"))
    fig.update_layout(template="plotly_dark",height=500,xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)
    section("Recent sessions")
    st.dataframe(df.tail(15).sort_values("Date",ascending=False),use_container_width=True,hide_index=True)

def traditional_page():
    heading("Rule-based research","Traditional strategies","Seven technical-rule baselines tested across the six-stock universe.")
    df=traditional_metrics()
    if df.empty:st.warning("Missing traditional_strategy_research/equal_weight_portfolio_metrics.csv");return
    section("Equal-weight six-stock portfolio · full sample")
    d=fmt_metrics(df)
    show_table(d)
    if {"strategy","total_return"}.issubset(df.columns):
        fig=px.bar(df.sort_values("total_return"),x="strategy",y="total_return",color="total_return",title="Strategy total return")
        fig.update_layout(template="plotly_dark",height=430,xaxis_tickangle=-25,yaxis_tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)
    section("Risk and risk-adjusted performance")
    cols=[c for c in ["strategy","CAGR","sharpe_ratio","sortino_ratio","max_drawdown","annualized_volatility"] if c in df]
    show_table(fmt_metrics(df[cols]))
    section("Per-stock strategy breakdown")
    stock=traditional_stock_metrics()
    if not stock.empty:
        ticker=st.selectbox("Stock breakdown",sorted(stock.ticker.unique()),key="tradstock")
        show_table(fmt_metrics(stock[stock.ticker==ticker]))

def hybrid_trad_page():
    heading("Rule combinations","Hybrid traditional strategies","Compare indicator combinations against the standalone rule-based baselines.")
    df=hybrid_traditional_metrics()
    if df.empty:st.warning("Missing hybrid_traditional_research/equal_weight_portfolio_metrics.csv");return
    section("Equal-weight portfolio results")
    show_table(fmt_metrics(df))
    fig=px.bar(df.sort_values("total_return"),x="strategy",y="total_return",color="total_return",title="Hybrid strategy total return")
    fig.update_layout(template="plotly_dark",height=400,xaxis_tickangle=-20,yaxis_tickformat=".0%")
    st.plotly_chart(fig,use_container_width=True)
    section("Per-stock comparison")
    stock=hybrid_traditional_stock_metrics()
    if not stock.empty:
        ticker=st.selectbox("Stock",sorted(stock.ticker.unique()),key="hybtradstock")
        show_table(fmt_metrics(stock[stock.ticker==ticker]))

def ml_page():
    heading("Predictive modelling","ML & sequence experiments","Classification quality and trading results from technical-indicator, recurrent and ensemble experiments.")
    tab1,tab2,tab3=st.tabs(["Technical-indicator ML","LSTM / GRU","Hybrid ensemble"])
    with tab1:
        d=technical_ml_metrics(); p=technical_ml_portfolio_metrics(); s=technical_ml_strategy_metrics()
        section("Model prediction metrics · per stock")
        show_table(fmt_metrics(d))
        section("Equal-weight six-stock test portfolio")
        show_table(fmt_metrics(p))
        if not p.empty:
            fig=px.bar(p.sort_values("total_return"),x="model",y="total_return",color="model",title="Technical ML model portfolio test return")
            fig.update_layout(template="plotly_dark",height=380,yaxis_tickformat=".0%")
            st.plotly_chart(fig,use_container_width=True)
        section("Trading metrics by stock and model")
        if not s.empty:
            ticker=st.selectbox("Stock",sorted(s.ticker.unique()),key="mlstock")
            show_table(fmt_metrics(s[s.ticker==ticker]))
    with tab2:
        d=sequence_metrics()
        section("Sequence model metrics by prediction horizon")
        show_table(fmt_metrics(d))
        st.caption("Metrics are shown for the saved sequence-model evaluation; horizon-specific results should be interpreted separately.")
    with tab3:
        d=ensemble_metrics(); stock=ensemble_stock_metrics()
        section("Pooled model comparison by horizon")
        show_table(fmt_metrics(d))
        section("Per-stock model metrics")
        if not stock.empty:
            ticker=st.selectbox("Stock",sorted(stock.ticker.unique()),key="ensstock")
            show_table(fmt_metrics(stock[stock.ticker==ticker]))

def hybrid_ml_page():
    heading("Pooled model strategy","Hybrid ML engine","Compare the selected hybrid ML strategy with an RSI recovery baseline and buy-and-hold on the same test window.")
    df=hybrid_ml_metrics()
    if df.empty:st.warning("Missing hybrid_ml_engine/test_strategy_metrics.csv");return
    show_table(fmt_metrics(df))
    if {"strategy","total_return"}.issubset(df.columns):
        fig=px.bar(df,x="strategy",y="total_return",color="strategy",title="Same-window test-period return comparison")
        fig.update_layout(template="plotly_dark",height=420,yaxis_tickformat=".0%")
        st.plotly_chart(fig,use_container_width=True)
    row=df[df.strategy.str.contains("Hybrid ML",case=False)]
    if not row.empty:
        r=row.iloc[0]
        section("Model diagnostics")
        c=st.columns(4)
        with c[0]:metric("Selected model",str(r.get("model","—")))
        with c[1]:metric("Test ROC-AUC",num(r.get("test_auc")))
        with c[2]:metric("Balanced accuracy",num(r.get("test_balanced_accuracy")))
        with c[3]:metric("Validation threshold",num(r.get("selected_threshold")))

def portfolio_page():
    heading("Allocation and risk","Portfolio & explainability","Saved portfolio allocation, performance and feature-attribution results.")
    weights,perf,returns=portfolio_data()
    t1,t2,t3=st.tabs(["Portfolio","Risk / returns","Explainable AI"])
    with t1:
        if weights.empty:st.info("No portfolio_weights.csv found.")
        else:
            show_table(weights)
            numeric=weights.select_dtypes("number")
            if not numeric.empty:
                vals=numeric.iloc[-1].dropna()
                vals=vals[vals>0]
                if not vals.empty:
                    fig=px.pie(values=vals.values,names=vals.index,hole=.55,title="Saved portfolio weights")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig,use_container_width=True)
    with t2:
        if perf.empty and returns.empty:st.info("No portfolio performance/returns saved.")
        else:
            show_table(perf if not perf.empty else returns)
            if not returns.empty:
                x=next((c for c in returns if c.lower() in ("date","datetime")),returns.columns[0])
                y=next((c for c in returns.select_dtypes("number") if c.lower() not in ("unnamed: 0",)),None)
                if y:
                    fig=px.line(returns,x=x,y=y,title="Portfolio daily returns")
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig,use_container_width=True)
    with t3:
        shap=shap_importance()
        if shap.empty:st.info("No SHAP importance files found.")
        else:
            feature=next((c for c in shap if c.lower() in ("feature","feature_name")),None)
            val=next((c for c in shap if "importance" in c.lower()),None)
            show_table(shap)
            if feature and val:
                fig=px.bar(shap.sort_values(val).tail(20),x=val,y=feature,color="Model" if "Model" in shap else None,orientation="h",title="Top SHAP features")
                fig.update_layout(template="plotly_dark",height=550)
                st.plotly_chart(fig,use_container_width=True)

def conclusions():
    heading("Research interpretation","Findings & limitations","A recruiter-facing summary of what the experiments actually show.")
    t=traditional_metrics(); h=hybrid_traditional_metrics(); ml=technical_ml_portfolio_metrics(); hm=hybrid_ml_metrics(); seq=sequence_metrics(); ens=ensemble_metrics()
    section("What the saved experiments indicate")
    if not t.empty:
        r=t.sort_values("sharpe_ratio",ascending=False).iloc[0]
        st.markdown(f"- **Traditional baseline:** {r['strategy']} led the equal-weight portfolio by Sharpe ({num(r['sharpe_ratio'])}) with total return {pct(r['total_return'])}.")
    if not h.empty:
        r=h.sort_values("total_return",ascending=False).iloc[0]
        st.markdown(f"- **Hybrid rule combinations:** {r['strategy']} was the strongest saved hybrid traditional strategy ({pct(r['total_return'])}); combinations did not consistently improve on simpler baselines.")
    if not ml.empty:
        r=ml.sort_values("sharpe_ratio",ascending=False).iloc[0]
        st.markdown(f"- **Technical-indicator ML:** {r['model']} was the strongest saved equal-weight test portfolio by Sharpe ({num(r['sharpe_ratio'])}), with test return {pct(r['total_return'])}.")
    if not hm.empty:
        r=hm[hm.strategy.str.contains("Hybrid ML",case=False)]
        if not r.empty:
            x=r.iloc[0]
            st.markdown(f"- **Hybrid ML trading engine:** test return {pct(x['total_return'])}, ROC-AUC {num(x.get('test_auc'))}; compare against its same-window RSI and buy-and-hold baselines.")
    if not seq.empty:
        best=seq.sort_values("roc_auc",ascending=False).iloc[0]
        st.markdown(f"- **Sequence models:** best recorded ROC-AUC was {num(best['roc_auc'])} for {best['model']} at {best['horizon']}-session horizon.")
    if not ens.empty:
        best=ens.sort_values("roc_auc",ascending=False).iloc[0]
        st.markdown(f"- **Hybrid ensemble:** best pooled ROC-AUC was {num(best['roc_auc'])} for {best['model']} at {best['horizon']} horizon.")
    section("Methodology and limitations")
    st.markdown("""- Historical backtests are not live performance or investment advice.
- Model metrics and strategy returns measure different things; a strong classifier does not guarantee a profitable strategy.
- Full-period traditional results must not be ranked directly against shorter ML holdout returns.
- Costs, slippage, execution timing, sample selection and market regime affect outcomes.
- Treat weak or below-random test metrics as valid negative research findings; do not tune against the final test set.""")
    st.markdown('<div class="info">The dashboard reads existing CSV outputs only. It does not train models, fabricate missing metrics or modify the underlying research artifacts.</div>',unsafe_allow_html=True)

pages={"Research Overview":overview,"Market Explorer":market_page,"Traditional Strategies":traditional_page,
       "Hybrid Strategies":hybrid_trad_page,"ML & Sequence Models":ml_page,
       "Hybrid ML Engine":hybrid_ml_page,"Portfolio & Explainability":portfolio_page,
       "Research Conclusions":conclusions}
with st.sidebar:
    st.markdown("## ◈ Quant<span style='color:#42dfcf'>Edge</span>",unsafe_allow_html=True)
    st.caption("QUANTITATIVE RESEARCH TERMINAL")
    st.markdown("---")
    page=st.radio("WORKSPACE",list(pages))
    st.markdown("---")
    st.caption("NSE UNIVERSE")
    st.caption("RELIANCE · HDFCBANK · ICICIBANK")
    st.caption("INFY · TCS · SBIN")
    st.markdown("---")
    st.caption("Saved research · No live trading")
pages[page]()
