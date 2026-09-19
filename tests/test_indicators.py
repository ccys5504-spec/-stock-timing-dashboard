import numpy as np
import pandas as pd

from signals.indicators import (
    ADX_NO_TREND_THRESHOLD, build_signals, compute_adx, compute_atr,
)
from tests.helpers import make_ohlcv


def test_atr_constant_range_equals_range():
    dates = pd.bdate_range("2024-01-01", periods=60)
    df = pd.DataFrame({"Open": 100.0, "High": 105.0, "Low": 95.0, "Close": 100.0, "Volume": 1e6}, index=dates)
    atr = compute_atr(df).dropna()
    assert np.allclose(atr.iloc[-1], 10.0)


def test_adx_high_in_trend_low_in_range():
    dates = pd.bdate_range("2022-01-03", periods=250)
    trend_close = pd.Series(np.linspace(100, 200, 250), index=dates)
    trend = pd.DataFrame({"Open": trend_close, "High": trend_close * 1.005,
                          "Low": trend_close * 0.995, "Close": trend_close}, index=dates)
    rng_close = pd.Series(100 + 3 * np.sin(np.arange(250) / 3), index=dates)
    ranging = pd.DataFrame({"Open": rng_close, "High": rng_close * 1.005,
                            "Low": rng_close * 0.995, "Close": rng_close}, index=dates)
    assert compute_adx(trend).dropna().iloc[-1] > ADX_NO_TREND_THRESHOLD
    assert compute_adx(ranging).dropna().iloc[-1] < compute_adx(trend).dropna().iloc[-1]


def test_indicators_use_only_past_data():
    """마지막 날을 잘라내도 그 전날까지의 지표/신호는 그대로여야 한다(미래 참조 금지)."""
    dates = pd.bdate_range("2022-01-03", periods=300)
    df = make_ohlcv(dates, seed=7)
    full = build_signals(df)
    part = build_signals(df.iloc[:-30])
    common = part.index
    cols = ["MA_SHORT", "MA_LONG", "RSI", "SCORE_TOTAL"]
    pd.testing.assert_frame_equal(full.loc[common, cols], part[cols], check_freq=False)
    assert (full.loc[common, "SIGNAL"] == part["SIGNAL"]).all()


def test_signal_labels_are_within_known_set():
    dates = pd.bdate_range("2022-01-03", periods=300)
    out = build_signals(make_ohlcv(dates, seed=11))
    assert set(out["SIGNAL"].unique()) <= {"매수", "매도", "관망"}
