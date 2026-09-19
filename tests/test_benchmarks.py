import pandas as pd
import pytest

from backtest.benchmarks import equal_weight_equity, summarize_equity


def _frame(values):
    idx = pd.bdate_range("2024-01-01", periods=len(values))
    return pd.DataFrame({"Close": values}, index=idx)


def test_equal_weight_is_average_of_daily_returns():
    data = {"A": _frame([100, 110, 121]), "B": _frame([100, 100, 100])}
    eq = equal_weight_equity(data, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"))
    # A는 +10%, +10% / B는 0% → 일평균 +5%, +5%
    assert eq.iloc[-1] == pytest.approx(1.05 * 1.05)


def test_equal_weight_handles_late_listing():
    a = _frame([100, 110, 121, 133.1])
    b = _frame([100, 100]).set_axis(a.index[2:])  # 3일째부터 상장
    eq = equal_weight_equity({"A": a, "B": b}, a.index[0], a.index[-1])
    assert len(eq) == 4 and eq.notna().all()


def test_equal_weight_empty_period_raises():
    with pytest.raises(ValueError):
        equal_weight_equity({"A": _frame([1, 2])}, pd.Timestamp("2030-01-01"), pd.Timestamp("2030-02-01"))


def test_summarize_equity_return_and_mdd():
    idx = pd.bdate_range("2024-01-01", periods=4)
    total, mdd = summarize_equity(pd.Series([1.0, 1.5, 0.9, 1.2], index=idx))
    assert total == pytest.approx(0.2)
    assert mdd == pytest.approx(0.9 / 1.5 - 1)


def test_pit_equal_weight_holds_within_month_and_handles_delisting():
    from backtest.benchmarks import pit_equal_weight_equity
    from tests.helpers import make_ohlcv

    dates = pd.bdate_range("2023-01-02", periods=520)
    a = make_ohlcv(dates, seed=1, drift=0.001, vol=0.01)
    b = make_ohlcv(dates[:400], seed=2, drift=0.001, vol=0.01)  # 400일째에 상장폐지
    shares = {"A": 1e6, "B": 1e6}
    cal = dates[300:]
    eq = pit_equal_weight_equity({"A": a, "B": b}, cal, top_n=2, shares=shares)
    assert len(eq) == len(cal)
    assert eq.iloc[0] == 1.0 and (eq > 0).all() and eq.notna().all()
    # 폐지 뒤에는 A 하나만 남아 A와 같은 방향으로 움직여야 한다(죽은 B가 계속 후보로 남지 않음)
    tail = eq.iloc[-40:]
    a_tail = a["Close"].reindex(cal).iloc[-40:]
    assert (tail.pct_change().dropna().round(6) == a_tail.pct_change().dropna().round(6)).mean() > 0.9
