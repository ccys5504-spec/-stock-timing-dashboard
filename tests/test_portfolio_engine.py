"""v2 포트폴리오 엔진의 순수 함수와 룩어헤드 방지 성질 검증."""
import pandas as pd
import pytest

from backtest.portfolio_engine import (
    RANK_CUTOFF_MULTIPLE, _inverse_vol_weights, _month_end_dates,
    _regime_multiplier, _select_with_hysteresis, run_portfolio_backtest,
)
from tests.helpers import make_ohlcv


def _ranked(codes):
    return pd.DataFrame({"코드": codes, "모멘텀점수": range(len(codes), 0, -1)})


def test_hysteresis_keeps_holdings_until_rank_passes_cutoff():
    ranked = _ranked([f"S{i}" for i in range(10)])
    # S4는 top_k=3 밖이지만 cutoff(6) 안 → 유지. S7은 cutoff 밖 → 제외.
    picked = _select_with_hysteresis(ranked, 3, 6, {"S4", "S7"})
    assert "S4" in picked and "S7" not in picked
    assert picked == ["S4", "S0", "S1"]


def test_hysteresis_empty_ranking_returns_empty():
    assert _select_with_hysteresis(pd.DataFrame(), 3, 6, {"A"}) == []


def test_rank_cutoff_scales_with_top_k():
    # 2026-09-20 점검서: rank_cutoff이 고정 상수(30)라 top_k를 바꾸면 의미가 달라지던 버그.
    assert RANK_CUTOFF_MULTIPLE == 2
    ranked = _ranked([f"S{i}" for i in range(40)])
    for top_k in (5, 15):
        inside, outside = f"S{top_k * 2 - 1}", f"S{top_k * 2}"
        picked = _select_with_hysteresis(ranked, top_k, top_k * RANK_CUTOFF_MULTIPLE, {inside, outside})
        assert inside in picked
        assert outside not in picked


def test_regime_multiplier_states():
    n = 260
    dates = pd.bdate_range("2020-01-01", periods=n)
    up = pd.DataFrame({"Close": [100 + i for i in range(n)]}, index=dates)
    assert _regime_multiplier(up) == 1.00  # 200일선 위 + 기울기 상승
    down = pd.DataFrame({"Close": [400 - i for i in range(n)]}, index=dates)
    assert _regime_multiplier(down) == 0.20  # 아래 + 기울기 하락
    assert _regime_multiplier(up.head(50)) == 1.0  # 데이터 부족 → 중립


def test_inverse_vol_weights_sum_and_cap():
    dates = pd.bdate_range("2023-01-02", periods=120)
    vols = [0.005, 0.02] + [0.05] * 8
    data = {f"S{i}": make_ohlcv(dates, seed=i, vol=v) for i, v in enumerate(vols)}
    w = _inverse_vol_weights(list(data), data, dates[-1], max_weight=0.2)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)
    assert max(w.values()) <= 0.2 + 1e-6
    assert w["S0"] >= w["S9"]  # 변동성이 낮을수록 비중이 크거나 같다(상한 때문에 같을 수 있음)


def test_inverse_vol_weights_ignores_missing_or_flat_stocks():
    dates = pd.bdate_range("2023-01-02", periods=120)
    flat = make_ohlcv(dates, seed=1, vol=0.0, drift=0.0)
    flat[["Open", "High", "Low", "Close"]] = 10_000.0
    ok = make_ohlcv(dates, seed=2, vol=0.02)
    w = _inverse_vol_weights(["FLAT", "OK", "GHOST"], {"FLAT": flat, "OK": ok}, dates[-1])
    assert set(w) == {"OK"}


def test_month_end_dates_picks_last_trading_day_of_each_month():
    cal = pd.bdate_range("2024-01-01", "2024-03-31")
    ends = _month_end_dates(cal)
    assert ends == {pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29"), pd.Timestamp("2024-03-29")}


# ---- 엔드투엔드(합성 데이터) ---------------------------------------------------

def _synthetic_world(n_days=700, n_stocks=12):
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    price_data = {
        f"{i:06d}": make_ohlcv(dates, seed=i, drift=0.0003 + 0.0001 * (i % 4), vol=0.018)
        for i in range(n_stocks)
    }
    market = {c: ("KOSPI" if int(c) % 2 == 0 else "KOSDAQ") for c in price_data}
    idx = make_ohlcv(dates, seed=999, drift=0.0004, vol=0.01, start_price=2500)
    return dates, price_data, market, {"KOSPI": idx, "KOSDAQ": idx}


def test_backtest_runs_with_positive_equity_and_logs_rebalances():
    dates, pdata, market, index = _synthetic_world()
    r = run_portfolio_backtest(pdata, market, index, dates[400], dates[-1], top_k=5)
    assert len(r.equity_curve) > 0
    assert (r.equity_curve > 0).all()
    assert r.holdings_log


def test_no_lookahead_future_data_does_not_change_past_equity():
    """끝 날짜 이후 데이터를 잘라내도 그 이전 자산곡선이 같아야 한다(룩어헤드 방지)."""
    dates, pdata, market, index = _synthetic_world()
    cut = dates[600]
    full = run_portfolio_backtest(pdata, market, index, dates[400], dates[-1], top_k=5)
    trunc_p = {c: d[d.index <= cut] for c, d in pdata.items()}
    trunc_i = {m: d[d.index <= cut] for m, d in index.items()}
    part = run_portfolio_backtest(trunc_p, market, trunc_i, dates[400], cut, top_k=5)
    pd.testing.assert_series_equal(
        full.equity_curve.loc[part.equity_curve.index], part.equity_curve,
        check_names=False, check_freq=False, rtol=1e-9,
    )


def test_higher_cost_never_improves_result():
    dates, pdata, market, index = _synthetic_world()
    args = (pdata, market, index, dates[400], dates[-1])
    low = run_portfolio_backtest(*args, top_k=5, round_trip_cost=0.003)
    high = run_portfolio_backtest(*args, top_k=5, round_trip_cost=0.010)
    assert high.total_return_pct <= low.total_return_pct + 1e-12


def test_empty_period_raises():
    dates, pdata, market, index = _synthetic_world()
    with pytest.raises(ValueError):
        run_portfolio_backtest(pdata, market, index, dates[-1] + pd.Timedelta(days=30),
                               dates[-1] + pd.Timedelta(days=60))


# ---- 실험 스위치(scripts/portfolio_ablation.py) --------------------------------

def test_switch_defaults_do_not_change_baseline():
    dates, pdata, market, index = _synthetic_world()
    args = (pdata, market, index, dates[400], dates[-1])
    base = run_portfolio_backtest(*args, top_k=5)
    explicit = run_portfolio_backtest(
        *args, top_k=5, use_regime=False, atr_initial_mult=2.5, atr_trail_mult=3.5, weighting="inverse_vol",
    )
    pd.testing.assert_series_equal(base.equity_curve, explicit.equity_curve)


def test_stops_off_means_no_stop_outs():
    dates, pdata, market, index = _synthetic_world()
    r = run_portfolio_backtest(
        pdata, market, index, dates[400], dates[-1], top_k=5, atr_initial_mult=None, atr_trail_mult=None,
    )
    assert r.num_stopped_out == 0


def test_invalid_weighting_rejected():
    dates, pdata, market, index = _synthetic_world()
    with pytest.raises(ValueError):
        run_portfolio_backtest(pdata, market, index, dates[400], dates[-1], weighting="magic")


def test_regime_filter_is_off_by_default():
    from backtest.portfolio_engine import USE_REGIME_FILTER
    assert USE_REGIME_FILTER is False
