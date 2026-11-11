import pandas as pd
import pytest

from strategy.exit_rules import DEFAULT_RULE, ExitRule, evaluate_holding, simulate_exit


def bars(rows, start="2026-01-05"):
    """rows: (open, high, low, close) 목록."""
    idx = pd.bdate_range(start, periods=len(rows))
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)


def test_floor_ratchets_with_peak():
    r = DEFAULT_RULE
    assert r.floor_return(0.0) == pytest.approx(-0.03)
    assert r.floor_return(0.049) == pytest.approx(-0.03)
    assert r.floor_return(0.05) == pytest.approx(0.05)    # 5%에 도달하면 매도선이 +5%로
    assert r.floor_return(0.099) == pytest.approx(0.05)
    assert r.floor_return(0.10) == pytest.approx(0.10)
    assert r.floor_return(0.35) == pytest.approx(0.20)    # 20% 이후에는 20%에 머문다


def test_initial_stop_at_minus_3_percent_intraday():
    b = bars([(100, 101, 99, 100), (100, 100, 96.5, 98)])  # 둘째 날 저가가 97(-3%)보다 낮음
    res = simulate_exit(b, 100.0)
    assert res.triggered and res.exit_index == 1 and res.exit_price == pytest.approx(97.0)
    assert res.exit_floor == pytest.approx(-0.03)


def test_gap_down_fills_at_open_not_at_stop_line():
    b = bars([(100, 101, 99.5, 100), (92, 93, 90, 91)])  # 시가 92 < 97
    res = simulate_exit(b, 100.0)
    assert res.exit_price == pytest.approx(92.0) and res.exit_index == 1


def test_no_exit_when_price_stays_above_stop():
    b = bars([(100, 102, 98, 101), (101, 103, 97.5, 102)])
    res = simulate_exit(b, 100.0)
    assert not res.triggered and res.peak_return == pytest.approx(0.03) and res.last_return == pytest.approx(0.02)


def test_reaching_5_percent_does_not_sell_but_falling_back_to_5_percent_does():
    b = bars([
        (100, 101, 99, 100),
        (101, 106, 101, 105),    # 고점 +6% -> 매도선이 +5%로 올라감(이날은 판정에 전날 고점 사용, 매도 없음)
        (105, 106, 104, 105),    # 저가 104 = +4% < +5% -> 매도선(105)에 닿아 매도
    ])
    res = simulate_exit(b, 100.0)
    assert res.triggered and res.exit_index == 2
    assert res.exit_price == pytest.approx(105.0) and res.exit_floor == pytest.approx(0.05)


def test_same_day_peak_does_not_raise_the_floor_for_that_day():
    # 하루 안에서 +6%까지 갔다가 -4%까지 빠지는 날: 그날의 매도선은 전날 고점 기준(-3%)이라 -3% 선에서 팔린 것으로 본다
    b = bars([(100, 106, 96, 100)])
    res = simulate_exit(b, 100.0)
    assert res.triggered and res.exit_price == pytest.approx(97.0) and res.exit_floor == pytest.approx(-0.03)


def test_higher_milestones_raise_the_floor_to_10_and_20():
    b = bars([
        (100, 111, 100, 110),     # 고점 +11% -> 매도선 +10%
        (110, 112, 108, 109),     # 저가 108 < 110 -> 매도선 110에서 매도
    ])
    res = simulate_exit(b, 100.0)
    assert res.exit_index == 1 and res.exit_price == pytest.approx(110.0) and res.exit_floor == pytest.approx(0.10)

    b2 = bars([(100, 125, 100, 124), (124, 126, 118, 119), (119, 120, 110, 111)])
    res2 = simulate_exit(b2, 100.0)
    assert res2.exit_index == 1 and res2.exit_price == pytest.approx(120.0) and res2.exit_floor == pytest.approx(0.20)


def test_floor_never_goes_down_after_a_pullback():
    b = bars([
        (100, 106, 100, 105),    # 매도선 +5%
        (105.6, 107, 105.5, 106),  # 시가·저가가 모두 매도선(105)보다 위 -> 유지
        (106, 106, 105.2, 105.4),
    ])
    res = simulate_exit(b, 100.0)
    assert not res.triggered and res.floor_return == pytest.approx(0.05)


def test_custom_rule():
    rule = ExitRule(stop_loss=0.05, milestones=(0.10,))
    b = bars([(100, 101, 96, 98)])
    assert not simulate_exit(b, 100.0, rule).triggered  # -4% 는 -5% 선 위
    assert simulate_exit(b, 100.0, ExitRule()).triggered  # 기본(-3%)이면 발동


def test_evaluate_holding_with_entry_date_tracks_the_peak_since_purchase():
    b = bars([(100, 112, 100, 111), (111, 113, 111, 112), (112, 112, 111, 111.5)], start="2026-02-02")
    older = pd.concat([bars([(50, 200, 50, 60)], start="2026-01-05"), b])  # 매수 전의 고점은 무시되어야 한다
    out = evaluate_holding(older, entry_price=100.0, entry_date="2026-02-02")
    assert not out["triggered"] and out["precise"]
    assert out["peak_return"] == pytest.approx(0.13)
    assert out["floor_return"] == pytest.approx(0.10) and out["floor_price"] == pytest.approx(110.0)
    assert out["last_return"] == pytest.approx(0.115)


def test_evaluate_holding_without_entry_date_uses_only_latest_bar_and_says_so():
    df = bars([(100, 101, 99, 100), (100, 108, 100, 107)])
    out = evaluate_holding(df, entry_price=100.0, entry_date=None)
    assert out["precise"] is False and not out["triggered"]
    assert out["last_return"] == pytest.approx(0.07)


def test_evaluate_holding_reports_trigger_date_and_validates_input():
    b = bars([(100, 101, 99, 100), (100, 100, 95, 96)], start="2026-03-02")
    out = evaluate_holding(b, 100.0, "2026-03-02")
    assert out["triggered"] and out["exit_date"] == pd.Timestamp("2026-03-03") and out["exit_price"] == pytest.approx(97.0)
    with pytest.raises(ValueError):
        evaluate_holding(b, 100.0, "2027-01-01")  # 매수일 이후 시세 없음
    with pytest.raises(ValueError):
        simulate_exit(b, 0.0)


def test_opening_exactly_on_the_floor_counts_as_touching_it():
    b = bars([(100, 106, 100, 105), (105, 106, 105, 105.5)])  # 둘째 날 시가 105 = 매도선 +5%
    res = simulate_exit(b, 100.0)
    assert res.triggered and res.exit_index == 1 and res.exit_price == pytest.approx(105.0)


def test_halted_bar_with_zero_prices_is_ignored_not_treated_as_a_crash():
    b = bars([(100, 101, 99, 100), (0, 0, 0, 100), (100, 102, 99, 101)])  # 둘째 날은 거래정지(가격 0으로 채워짐)
    res = simulate_exit(b, 100.0)
    assert not res.triggered and res.last_return == pytest.approx(0.01)
