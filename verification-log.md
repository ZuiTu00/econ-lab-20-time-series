# Verification Log: AI-Assisted Streamlit App

**Course:** ECON 5200, Lab 20
**Date:** April 2026
**AI tool used:** Claude (Anthropic)

This log documents the audit trail for the AI-assisted portion of Lab 20. It records the prompt, AI output, human modifications with justification, and verification against the notebook's ground-truth.

---

## 1. P.R.I.M.E. Prompt Used

[Prep] Act as an expert Python Data Scientist specializing in time series analysis, FRED API, and production ML systems.

[Request] Completed a diagnosis-first lab (broken STL on multiplicative data, misspecified ADF, MSTL, block bootstrap, decompose.py module). Build an interactive Streamlit app for FRED series with decomposition + stationarity + breaks + bootstrap CI.

[Iterate] Use streamlit, plotly, fredapi, statsmodels, ruptures. Handle missing data and frequency detection automatically.

[Mechanism Check] Explain inline: why block bootstrap preserves autocorrelation (i.i.d. destroys it), how MSTL iteratively removes seasonals, why PELT penalty controls bias-variance.

[Evaluate] Discuss sensitivity of decomposition to parameter choices.

---

## 2. AI Output Summary

AI generated a Streamlit app (~200 lines) with sidebar controls (FRED ID, date range, method, period, robust, PELT penalty, bootstrap toggle), FRED data fetch with caching, 4-panel Plotly decomposition, 2x2 stationarity results, PELT breakpoints, and optional block bootstrap CI. The app imports my existing decompose.py rather than re-implementing logic.

---

## 3. Human Modifications and Justifications

### Modification 1: Forced KPSS nlags=auto

AI's initial draft used kpss(series, regression=ct) without specifying nlags. This defaults to a small value that underestimates long-run variance for persistent series like GDP, inflating Type I error. I changed it to nlags=auto (Schwert 1989 rule), matching notebook Cell [10]. Test: GDPC1 level returns ADF p=0.9617, KPSS p=0.0100 in both notebook and app.

### Modification 2: Default PELT penalty = 3.0, not 10.0

AI copied the notebook's pen=10, but running on GDP growth detected zero real breaks - only a spurious sample-end index. The Great Moderation (~1984, Stock & Watson 2003) should be detectable. I changed default to pen=3.0, which correctly detects 1985-04-01. Slider now ranges 1.0-20.0 so users can explore the bias-variance tradeoff.

### Modification 3: MSTL secondary period hardcoded as period times 7

AI exposed MSTL with periods [period, period*7]. Works for daily-to-weekly (7=days/week) but nonsensical for quarterly (4 becomes 28 quarters = 7 years, no economic meaning). I flagged this as a design limitation rather than rewriting - MSTL genuinely only suits hourly/sub-daily data.

### Modification 4: Bootstrap block_size capped at 24

AI's slider allowed block_size up to 50. For n=263 quarterly GDP obs, block_size >= 24 leaves ~11 distinct start positions - near-deterministic replay. I capped at 24 (6 years), generous vs. the n^(1/3) ~ 6 rule of thumb.

### Modification 5: Bootstrap default off

AI enabled bootstrap by default. 100-500 STL fits takes 30-120s - bad UX. Changed to off with explicit checkbox and spinner.

---

## 4. Verification Against Notebook Ground Truth

| Check | Notebook | App | Match |
|---|---|---|---|
| GDPC1 level ADF p (ct) | 0.9617 | 0.9617 | yes |
| GDPC1 level KPSS p (ct) | 0.0100 | 0.0100 | yes |
| GDPC1 level verdict | non-stationary | non-stationary | yes |
| GDPC1 diff verdict | contradictory | contradictory | yes |
| GDP growth breaks @ pen=3 | [1985-04-01] | [1985-04-01] | yes |

All numerical results match to 4 decimal places.

---

## 5. Remaining Limitations

1. MSTL hardcoding (Mod 3): still a design flaw.
2. PELT model=rbf hardcoded. For mean-shift-only cases, l2 would be faster.
3. No holiday/calendar adjustment. Known limitation of lab scope.

---

## 6. Meta-Reflection on AI Assistance

The AI's initial draft was ~85% correct but contained three problem categories requiring domain expertise to detect: silent defaults (KPSS nlags), parameter transplant (pen=10), and unjustified generalization (MSTL second period). The lab's diagnosis-first design was essential to catching these - without the notebook's ground-truth runs, Modifications 1 and 2 would have produced a plausible-looking but methodologically inferior app. This is the core lesson of "Foundations First, Expansion Second."
