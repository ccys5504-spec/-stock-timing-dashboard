import pandas as pd

from strategy.live_portfolio import STATUS_HELD, STATUS_NOT_HELD, build_target_portfolio, dropped_holdings
from tests.helpers import make_ohlcv


def _world(n=30):
    dates = pd.bdate_range("2023-01-02", periods=450)
    data = {f"{i:06d}": make_ohlcv(dates, seed=i, drift=0.0002 + 0.0001 * (i % 6), vol=0.015, volume=2e6 + i * 1e5)
            for i in range(n)}
    names = {c: f"종목{c}" for c in data}
    markets = {c: "KOSPI" for c in data}
    return data, names, markets


def test_target_portfolio_shape_weights_and_stop_below_price():
    data, names, markets = _world()
    table, as_of = build_target_portfolio(data, names, markets, top_k=10, universe_top_n=20)
    assert as_of == max(df.index[-1] for df in data.values())
    assert 0 < len(table) <= 10
    assert abs(table["목표비중(%)"].sum() - 100) < 1.0  # 반올림 오차 정도
    assert (table["목표비중(%)"] <= 12.0 + 1e-6).all() or len(table) < 9  # 종목 수가 적으면 상한이 못 지켜질 수 있음
    assert (table["손절참고가"] < table["현재가"]).all()
    assert table["순위"].is_monotonic_increasing
    assert set(table["상태"]) == {STATUS_NOT_HELD}


def test_holdings_are_kept_when_still_within_cutoff_and_reported_when_dropped():
    data, names, markets = _world()
    base, _ = build_target_portfolio(data, names, markets, top_k=10, universe_top_n=25)
    keep = base["종목코드"].iloc[0]
    gone = "999999"  # 후보에 없는(시세가 없는) 보유 종목
    table, _ = build_target_portfolio(data, names, markets, held_codes={keep, gone}, top_k=10, universe_top_n=25)
    assert table.loc[table["종목코드"] == keep, "상태"].iloc[0] == STATUS_HELD
    assert dropped_holdings(table, {keep, gone}, names) == [(gone, gone)]


def test_empty_when_no_data():
    table, as_of = build_target_portfolio({}, {}, {})
    assert table.empty and as_of is None


def test_regime_multiplier_scales_weights_and_labels():
    from strategy.live_portfolio import current_regime
    data, names, markets = _world()
    full, _ = build_target_portfolio(data, names, markets, top_k=10, universe_top_n=25)
    half, _ = build_target_portfolio(data, names, markets, top_k=10, universe_top_n=25,
                                     regime={"KOSPI": (0.4, "약세")})
    assert abs(half["목표비중(%)"].sum() - 0.4 * full["목표비중(%)"].sum()) < 1.0
    dates = pd.bdate_range("2023-01-02", periods=300)
    up = pd.DataFrame({"Close": [100 + i for i in range(300)]}, index=dates)
    assert current_regime({"KOSPI": up})["KOSPI"][0] == 1.0
    assert current_regime({"KOSDAQ": pd.DataFrame()})["KOSDAQ"][0] == 1.0  # 지수 없으면 중립


def test_default_live_ranking_is_market_cap_order():
    data, names, markets = _world()
    marcap = {c: float(1000 - i) for i, c in enumerate(sorted(data))}  # 코드 순서대로 시총이 작아진다
    table, _ = build_target_portfolio(data, names, markets, top_k=3, marcap=marcap)
    assert list(table["종목코드"]) == sorted(data)[:3]
    mom, _ = build_target_portfolio(data, names, markets, top_k=3, ranking="momentum")
    assert len(mom) == 3
