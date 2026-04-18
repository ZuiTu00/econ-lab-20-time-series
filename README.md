# econ-lab-20-time-series
# Lab 20 — Time Series Diagnostics & Advanced Decomposition

**Course:** ECON 5200 — Causal Machine Learning & Applied Analytics
**Topic:** Time Series I — Trends, Stationarity, Structural Breaks

This repository contains a diagnosis-first workflow for decomposing
macroeconomic time series, testing stationarity, quantifying trend
uncertainty via block bootstrap, and detecting structural regimes.

## Key findings

1. **Additive STL on multiplicative data is biased.** Log-transform
   fix reduces seasonal amplitude ratio from ~1.4× to 0.91 (ideal=1.0).
2. **ADF with `regression='n'` on trending GDP is invalid.** Test
   statistic falls on the wrong side of the rejection region
   (+8.39 vs. required <-1.94). Fixed with `regression='ct'` and
   confirmed by 2×2 decision table (ADF + KPSS).
3. **Block bootstrap CI is nearly 2× wider in recessions.** Width
   at 2008Q4 = 0.0106 vs. 2019Q4 = 0.0056 — reflecting heteroskedastic
   residual variance that i.i.d. bootstrap would erase.
4. **PELT detects the Great Moderation.** With `pen=3`, a
   breakpoint at 1985Q2 is detected — consistent with Stock &
   Watson (2003).

## How to Reproduce

```bash
git clone <this-repo>
cd econ-lab-20-time-series
pip install -r requirements.txt
export FRED_API_KEY=<your_key>   # free from https://fred.stlouisfed.org
jupyter notebook notebooks/lab_20_time_series.ipynb
```

For the interactive Streamlit app:
```bash
streamlit run app.py
```

## Repository Structure
