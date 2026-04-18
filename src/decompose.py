"""
decompose.py - Time Series Decomposition & Diagnostics Module
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
import ruptures as rpt
from typing import Optional


def run_stl(
    series: pd.Series,
    period: int = 12,
    log_transform: bool = True,
    robust: bool = True
):
    """Apply STL decomposition with optional log-transform."""
    work = series.dropna().copy()
    if work.index.freq is None:
        inferred = pd.infer_freq(work.index)
        if inferred is not None:
            work.index.freq = inferred
    if log_transform:
        if (work <= 0).any():
            raise ValueError("log_transform=True requires strictly positive values.")
        work = np.log(work)
    return STL(work, period=period, robust=robust).fit()


def test_stationarity(
    series: pd.Series,
    alpha: float = 0.05,
    regression: str = 'ct'
) -> dict:
    """Run ADF + KPSS and return the 2x2 decision table verdict."""
    x = series.dropna()
    adf_stat, adf_p, _, _, _, _ = adfuller(x, autolag='AIC', regression=regression)
    kpss_stat, kpss_p, _, _ = kpss(x, regression=regression, nlags='auto')
    adf_rej = adf_p < alpha
    kpss_rej = kpss_p < alpha
    if adf_rej and not kpss_rej:
        verdict = 'stationary'
    elif not adf_rej and kpss_rej:
        verdict = 'non-stationary'
    elif adf_rej and kpss_rej:
        verdict = 'contradictory'
    else:
        verdict = 'inconclusive'
    return {
        'adf_stat': float(adf_stat),
        'adf_p': float(adf_p),
        'kpss_stat': float(kpss_stat),
        'kpss_p': float(kpss_p),
        'verdict': verdict,
    }


def detect_breaks(
    series: pd.Series,
    pen: float = 3.0
) -> list:
    """Detect structural breaks using the PELT algorithm."""
    x = series.dropna()
    algo = rpt.Pelt(model='rbf').fit(x.values)
    idxs = algo.predict(pen=pen)
    return [x.index[i] for i in idxs if i < len(x)]


if __name__ == '__main__':
    print('decompose.py loaded successfully.')
