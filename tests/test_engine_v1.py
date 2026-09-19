"""v1 백테스트 엔진(backtest/engine.py) — 체결 시점과 손절 규칙이 다시 틀어지지 않게 고정.

2026-09-15에 "신호가 뜬 당일 종가에 즉시 체결"하던 치명적 오류를 고쳤다. 아래 테스트는
그 원칙(신호 다음 거래일 시가 체결, 손절은 전일 확정 기준선 vs 당일 저가, 갭하락은
시가로 체결)이 유지되는지 확인한다.
"""
import pandas as pd
import pytest

from backtest.engine import ROUND_TRIP_COST, run_backtest
from tests.helpers import make_signal_frame


def test_signal_fills_at_next_day_open_not_same_day_close():
    # 신호는 idx1(매수)·idx3(매도)에 뜬다 → 체결은 각각 idx2·idx4의 '시가'여야 한다
    df = make_signal_frame(
        opens=[100, 101, 110, 111, 120, 121],
        lows=[99, 100, 109, 110, 119, 120],
        closes=[100, 105, 112, 115, 118, 121],
        signals=["관망", "매수", "관망", "매도", "관망", "관망"],
    )
    result = run_backtest(df)
    trade = result.trades[0]
    assert trade.buy_date == df.index[2]
    assert trade.buy_price == 110  # 신호가 뜬 idx1의 종가(105)도, 당일 시가(101)도 아님
    assert trade.sell_date == df.index[4]
    assert trade.sell_price == 120


def test_signal_on_last_day_is_never_executed():
    df = make_signal_frame(
        opens=[100, 100, 100], lows=[99, 99, 99], closes=[100, 100, 100],
        signals=["관망", "관망", "매수"],
    )
    result = run_backtest(df)
    assert result.trades == []
    assert result.equity_curve.iloc[-1] == pytest.approx(1.0)


def test_repeated_buy_signals_while_holding_open_only_one_trade():
    df = make_signal_frame(
        opens=[100] * 6, lows=[99] * 6, closes=[100] * 6,
        signals=["매수", "매수", "매수", "관망", "관망", "관망"],
    )
    assert len(run_backtest(df).trades) == 1


def test_sell_signal_without_position_is_ignored():
    df = make_signal_frame(
        opens=[100] * 4, lows=[99] * 4, closes=[100] * 4,
        signals=["매도", "매도", "관망", "관망"],
    )
    assert run_backtest(df).trades == []


def test_gap_down_stop_fills_at_open_not_at_stop_price():
    # idx0 신호 → idx1 시가 100에 매수. 10% 손절선=90. idx2는 시가 80으로 갭하락 → 80에 청산
    df = make_signal_frame(
        opens=[100, 100, 80, 80], lows=[100, 100, 78, 78], closes=[100, 100, 82, 82],
        signals=["매수", "관망", "관망", "관망"],
    )
    trade = run_backtest(df, stop_loss_pct=0.10).trades[0]
    assert trade.stopped_out is True
    assert trade.sell_date == df.index[2]
    assert trade.sell_price == 80  # 손절선(90)이 아니라 실제로 열린 가격


def test_intraday_stop_fills_at_stop_price():
    # 시가(95)는 손절선(90) 위, 저가(88)가 손절선을 건드림 → 손절선 가격 90에 체결
    df = make_signal_frame(
        opens=[100, 100, 95, 95], lows=[100, 100, 88, 88], closes=[100, 100, 92, 92],
        signals=["매수", "관망", "관망", "관망"],
    )
    trade = run_backtest(df, stop_loss_pct=0.10).trades[0]
    assert trade.sell_price == pytest.approx(90.0)


def test_stop_not_triggered_when_low_stays_above_line():
    df = make_signal_frame(
        opens=[100, 100, 96, 96], lows=[100, 100, 91, 91], closes=[100, 100, 95, 95],
        signals=["매수", "관망", "관망", "관망"],
    )
    trade = run_backtest(df, stop_loss_pct=0.10).trades[0]
    assert trade.stopped_out is False and trade.sell_price is None


def test_trailing_stop_uses_previous_close_peak_and_todays_low():
    # idx1 시가 100 매수 → 종가 110, 130으로 오르며 고점 130. idx3에 저가 105가
    # 고점 대비 15% 트레일링선(130×0.85=110.5)을 건드림 → 시가(125)와 선(110.5) 중 낮은 110.5에 체결
    df = make_signal_frame(
        opens=[100, 100, 118, 125], lows=[100, 100, 120, 105], closes=[100, 110, 130, 112],
        signals=["매수", "관망", "관망", "관망"],
    )
    trade = run_backtest(df, trailing_stop_pct=0.15, exit_on_signal=False).trades[0]
    assert trade.trailing_stopped_out is True
    assert trade.sell_date == df.index[3]
    assert trade.sell_price == pytest.approx(110.5)


def test_trailing_stop_peak_ignores_todays_high_close():
    # 당일 종가로 고점을 올린 뒤 '같은 날' 저가로 이탈을 판정하면 안 된다(하루 안의 고가/저가
    # 순서를 일봉으로는 알 수 없음). idx2 종가 130이 고점이 되지만 그 기준선은 idx3부터 적용.
    df = make_signal_frame(
        opens=[100, 100, 118, 128], lows=[100, 100, 100, 128], closes=[100, 110, 130, 129],
        signals=["매수", "관망", "관망", "관망"],
    )
    # idx2 저가 100은 '전일(idx1) 종가 고점 110' 기준선 93.5보다 높으므로 청산되면 안 됨
    trade = run_backtest(df, trailing_stop_pct=0.15, exit_on_signal=False).trades[0]
    assert trade.sell_price is None


def test_trade_return_and_equity_use_same_cost_formula():
    df = make_signal_frame(
        opens=[100, 100, 100, 120, 120], lows=[99, 99, 99, 119, 119],
        closes=[100, 100, 110, 120, 120],
        signals=["매수", "관망", "매도", "관망", "관망"],
    )
    result = run_backtest(df)
    trade = result.trades[0]
    expected = (trade.sell_price / trade.buy_price) * (1 - ROUND_TRIP_COST) - 1
    assert trade.return_pct == pytest.approx(expected)
    # 자산곡선의 최종 수익률도 거래별 수익률과 같아야 한다(공식 불일치 재발 방지)
    assert result.total_return_pct == pytest.approx(expected)


def test_exit_on_signal_false_ignores_sell_signals():
    df = make_signal_frame(
        opens=[100] * 5, lows=[99] * 5, closes=[100] * 5,
        signals=["매수", "관망", "매도", "관망", "관망"],
    )
    trade = run_backtest(df, exit_on_signal=False).trades[0]
    assert trade.sell_price is None


def test_adaptive_exit_holds_through_sell_signal_in_bull_regime_only():
    base = dict(
        opens=[100] * 5, lows=[99] * 5, closes=[100] * 5,
        signals=["매수", "관망", "매도", "관망", "관망"],
    )
    bull = make_signal_frame(**base)
    bull["IS_BULL_REGIME"] = True
    assert run_backtest(bull, adaptive_exit=True).trades[0].sell_price is None

    bear = make_signal_frame(**base)
    bear["IS_BULL_REGIME"] = False
    assert run_backtest(bear, adaptive_exit=True).trades[0].sell_price == 100


def test_exposure_and_ratio_metrics():
    df = make_signal_frame(
        opens=[100, 101, 110, 111, 120, 121], lows=[99, 100, 109, 110, 119, 120],
        closes=[100, 105, 112, 115, 118, 121],
        signals=["관망", "매수", "관망", "매도", "관망", "관망"],
    )
    result = run_backtest(df)
    # 보유일: idx2, idx3 (idx4 시가에 청산 → 그날 종가 시점엔 미보유) → 6일 중 2일
    assert result.exposure_pct == pytest.approx(2 / 6)
    assert result.num_trades == 1
    assert result.win_rate_pct == 1.0

    flat = make_signal_frame([100] * 5, [99] * 5, [100] * 5, ["관망"] * 5)
    flat_result = run_backtest(flat)
    assert flat_result.sharpe_ratio is None  # 변동성 0이면 계산 불가
    assert flat_result.sortino_ratio is None
    assert flat_result.exposure_pct == 0.0
