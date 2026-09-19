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


def test_rank_universe_drops_delisted_stale_stock_and_keeps_it_before_delisting():
    dates = pd.bdate_range("2023-01-02", periods=500)
    alive = make_ohlcv(dates, seed=1, drift=0.001, vol=0.005)
    dead = make_ohlcv(dates[:300], seed=2, drift=0.003, vol=0.005)  # 300일째에 상장폐지
    data = {"ALIVE": alive, "DEAD": dead}
    before = rank_universe(data, dates[299])
    after = rank_universe(data, dates[-1])
    assert "DEAD" in before["코드"].tolist()  # 폐지 전에는 후보 (생존편향 제거의 핵심)
    assert "DEAD" not in after["코드"].tolist()  # 폐지 후에는 죽은 종목이 후보로 남지 않는다


def test_rank_universe_top_n_uses_trading_value_at_that_date():
    dates = pd.bdate_range("2023-01-02", periods=400)
    big = make_ohlcv(dates, seed=1, drift=0.0005, vol=0.005, volume=5e6)
    mid = make_ohlcv(dates, seed=2, drift=0.002, vol=0.005, volume=2e6)
    small = make_ohlcv(dates, seed=3, drift=0.003, vol=0.005, volume=1e6)
    ranked = rank_universe({"BIG": big, "MID": mid, "SMALL": small}, dates[-1], top_n=2)
    assert set(ranked["코드"]) == {"BIG", "MID"}  # 거래대금 상위 2개만, SMALL은 모멘텀이 제일 세도 제외
