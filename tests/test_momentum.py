import pandas as pd

from strategy.momentum import momentum_score, rank_universe, select_top_k
from tests.helpers import make_ohlcv


def _series(values, end="2024-12-31"):
    idx = pd.bdate_range(end=end, periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def test_momentum_none_when_history_too_short():
    assert momentum_score(_series([100.0] * 100), pd.Timestamp("2024-12-31")) is None


def test_momentum_ignores_data_after_as_of():
    close = _series([100 + i * 0.1 for i in range(400)])
    as_of = close.index[-50]
    truncated = close[close.index <= as_of]
    assert momentum_score(close, as_of) == momentum_score(truncated, as_of)


def test_momentum_higher_for_stronger_uptrend():
    weak = _series([100 + i * 0.05 for i in range(400)])
    strong = _series([100 + i * 0.5 for i in range(400)])
    as_of = weak.index[-1]
    assert momentum_score(strong, as_of) > momentum_score(weak, as_of)


def test_rank_universe_orders_and_filters_illiquid():
    dates = pd.bdate_range("2023-01-02", periods=400)
    strong = make_ohlcv(dates, seed=1, drift=0.002, vol=0.005)
    weak = make_ohlcv(dates, seed=2, drift=0.0001, vol=0.005)
    illiquid = make_ohlcv(dates, seed=3, drift=0.003, vol=0.005, volume=1.0)
    ranked = rank_universe({"STRONG": strong, "WEAK": weak, "ILLIQ": illiquid}, dates[-1])
    assert ranked["코드"].tolist() == ["STRONG", "WEAK"]  # 거래대금 미달 종목은 제외
    assert select_top_k(ranked, 1) == ["STRONG"]


def test_select_top_k_empty():
    assert select_top_k(pd.DataFrame(), 5) == []
