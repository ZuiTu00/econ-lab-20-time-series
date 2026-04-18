"""
app.py - Streamlit app for interactive time series diagnostics.

P.R.I.M.E. PROMPT USED TO SCAFFOLD THIS APP (verification trail):
----------------------------------------------------------------
[Prep] Act as an expert Python Data Scientist specializing in
time series analysis, FRED API, and production ML systems.

[Request] Completed a diagnosis-first lab (broken STL on
multiplicative data, misspecified ADF, MSTL, block bootstrap,
decompose.py module). Build: (1) extended decompose.py with
run_mstl and block_bootstrap_trend, (2) interactive Streamlit
app for FRED series with decomposition + stationarity + breaks
+ bootstrap CI.

[Iterate] streamlit, plotly, fredapi, statsmodels, ruptures.
Handle missing data and frequency detection automatically.

[Mechanism Check] Explain inline: why block bootstrap preserves
autocorrelation (i.i.d. destroys it), how MSTL iteratively
removes seasonals, why PELT penalty controls bias-variance.

[Evaluate] Discuss sensitivity of decomposition to parameter
choices.

See verification-log.md for the audit trail of what AI generated
vs. what I modified.
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fredapi import Fred
from statsmodels.tsa.seasonal import STL, MSTL
from decompose import run_stl, test_stationarity, detect_breaks


# ----------------- Setup -----------------
st.set_page_config(page_title="Time Series Diagnostics", layout="wide")
st.title("Time Series Decomposition & Diagnostics")
st.caption("Lab 20 - ECON 5200. Powered by FRED + decompose.py")

# FRED API key: prefer env var, fall back to sidebar input
api_key = os.environ.get("FRED_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("FRED API Key", type="password")
    if not api_key:
        st.warning("Enter a FRED API key in the sidebar to continue.")
        st.stop()

fred = Fred(api_key=api_key)


# ----------------- Sidebar controls -----------------
st.sidebar.header("Data")
series_id = st.sidebar.text_input("FRED series ID", value="GDPC1",
                                   help="e.g. GDPC1, RSXFSN, UNRATE, CPIAUCSL")
start_date = st.sidebar.date_input("Start date", value=pd.Timestamp("1960-01-01"))

st.sidebar.header("Decomposition")
method = st.sidebar.selectbox("Method", ["STL", "MSTL"])
log_transform = st.sidebar.checkbox("Log-transform", value=True,
                                     help="Use for multiplicative series")
period = st.sidebar.number_input("Seasonal period", min_value=2, value=4,
                                  help="4=quarterly, 12=monthly, 24=hourly")
robust = st.sidebar.checkbox("Robust (downweight outliers)", value=True)

st.sidebar.header("Structural breaks")
pen = st.sidebar.slider("PELT penalty", 1.0, 20.0, 3.0, step=0.5,
                         help="Lower = more breaks")

st.sidebar.header("Bootstrap CI")
run_bootstrap = st.sidebar.checkbox("Compute bootstrap CI (slow)", value=False)
n_bootstrap = st.sidebar.slider("Bootstrap replications", 50, 500, 100)
block_size = st.sidebar.slider("Block size", 2, 24, 8)


# ----------------- Fetch data -----------------
@st.cache_data
def fetch_series(sid: str, start):
    s = fred.get_series(sid, observation_start=start)
    s = s.dropna()
    s.index = pd.DatetimeIndex(s.index)
    inferred = pd.infer_freq(s.index)
    if inferred is not None:
        s.index.freq = inferred
    return s


try:
    series = fetch_series(series_id, start_date)
except Exception as e:
    st.error(f"Could not fetch {series_id}: {e}")
    st.stop()

st.subheader(f"Series: {series_id}")
st.write(f"Observations: {len(series)} | "
         f"Range: {series.index[0].date()} to {series.index[-1].date()}")


# ----------------- Decomposition -----------------
# MECHANISM: STL iteratively separates trend (via LOESS) and
# seasonal (via period averaging). MSTL extends this by
# backfitting multiple seasonal components -- subtracts each
# cycle before estimating the others, so they don't contaminate.
st.subheader("Decomposition")
try:
    if method == "STL":
        result = run_stl(series, period=int(period),
                         log_transform=log_transform, robust=robust)
        seasonal_series = result.seasonal
    else:  # MSTL
        work = np.log(series) if log_transform else series
        result = MSTL(work, periods=[int(period), int(period) * 7]).fit()
        seasonal_series = result.seasonal.iloc[:, 0]
except Exception as e:
    st.error(f"Decomposition failed: {e}")
    st.stop()

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=("Observed", "Trend", "Seasonal", "Residual"))
obs = np.log(series) if log_transform else series
fig.add_trace(go.Scatter(x=obs.index, y=obs.values, line=dict(width=1)), row=1, col=1)
fig.add_trace(go.Scatter(x=result.trend.index, y=result.trend.values,
                          line=dict(width=2, color="orange")), row=2, col=1)
fig.add_trace(go.Scatter(x=seasonal_series.index, y=seasonal_series.values,
                          line=dict(width=1, color="green")), row=3, col=1)
fig.add_trace(go.Scatter(x=result.resid.index, y=result.resid.values,
                          line=dict(width=1, color="red")), row=4, col=1)
fig.update_layout(height=700, showlegend=False)
st.plotly_chart(fig, use_container_width=True)


# ----------------- Stationarity -----------------
st.subheader("Stationarity (ADF + KPSS 2x2 decision)")
col1, col2 = st.columns(2)
with col1:
    st.markdown("**Level**")
    try:
        diag = test_stationarity(series, regression='ct')
        st.write(f"ADF p-value: `{diag['adf_p']:.4f}`")
        st.write(f"KPSS p-value: `{diag['kpss_p']:.4f}`")
        st.metric("Verdict", diag['verdict'].upper())
    except Exception as e:
        st.error(str(e))

with col2:
    st.markdown("**First difference**")
    try:
        diag_d = test_stationarity(series.diff().dropna(), regression='c')
        st.write(f"ADF p-value: `{diag_d['adf_p']:.4f}`")
        st.write(f"KPSS p-value: `{diag_d['kpss_p']:.4f}`")
        st.metric("Verdict", diag_d['verdict'].upper())
    except Exception as e:
        st.error(str(e))


# ----------------- Structural breaks -----------------
# MECHANISM: PELT minimizes [sum(segment_cost) + pen * n_breaks].
# Higher pen = fewer breaks (bias high, variance low).
# Lower pen = more breaks (may overfit noise).
st.subheader(f"Structural breaks (PELT, pen={pen})")
try:
    growth = series.pct_change().dropna() * 100
    breaks = detect_breaks(growth, pen=pen)
    msg = (f"Detected {len(breaks)} break(s): "
           + ", ".join(str(b.date()) for b in breaks)) if breaks else "None detected"
    st.write(msg)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=growth.index, y=growth.values,
                              mode="lines", line=dict(width=1), name="Growth (%)"))
    for b in breaks:
        fig2.add_vline(x=b, line=dict(color="red", dash="dash"))
    fig2.update_layout(height=300, showlegend=False,
                        title=f"{series_id} growth with breakpoints")
    st.plotly_chart(fig2, use_container_width=True)
except Exception as e:
    st.error(str(e))


# ----------------- Bootstrap CI -----------------
# MECHANISM: i.i.d. bootstrap shuffles residuals independently,
# destroying autocorrelation and underestimating uncertainty for
# macro series where shocks persist. Block bootstrap (Kunsch 1989)
# resamples contiguous blocks, preserving within-block dependence.
if run_bootstrap:
    st.subheader(f"Block bootstrap trend CI (n={n_bootstrap}, block={block_size})")
    with st.spinner("Running bootstrap..."):
        rng = np.random.default_rng(42)
        n = len(result.trend)
        orig_trend = result.trend.values
        orig_seasonal = (result.seasonal.iloc[:, 0].values
                         if method == "MSTL" else result.seasonal.values)
        orig_resid = result.resid.values

        boot_trends = np.zeros((n_bootstrap, n))
        for b in range(n_bootstrap):
            boot_resid = np.zeros(n)
            i = 0
            while i < n:
                start = rng.integers(0, n - block_size + 1)
                take = min(block_size, n - i)
                boot_resid[i:i + take] = orig_resid[start:start + take]
                i += take
            recon = pd.Series(orig_trend + orig_seasonal + boot_resid,
                              index=result.trend.index)
            recon.index.freq = result.trend.index.freq
            boot_trends[b, :] = STL(recon, period=int(period), robust=robust).fit().trend.values

        lower = np.percentile(boot_trends, 5, axis=0)
        upper = np.percentile(boot_trends, 95, axis=0)

    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=result.trend.index, y=upper, line=dict(width=0),
                              showlegend=False))
    fig3.add_trace(go.Scatter(x=result.trend.index, y=lower, line=dict(width=0),
                              fill='tonexty', fillcolor='rgba(52,152,219,0.3)',
                              name='90% CI'))
    fig3.add_trace(go.Scatter(x=result.trend.index, y=orig_trend,
                              line=dict(color='orange', width=2), name='Trend'))
    fig3.update_layout(height=400)
    st.plotly_chart(fig3, use_container_width=True)

st.caption("Built from Lab 20 / ECON 5200. See verification-log.md for AI audit trail.")
