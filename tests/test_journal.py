import json

import pandas as pd
import pytest

import data.fetch as fetch
from strategy.journal import (
    JournalError, apply_trade_to_holdings, daily_performance, realized_pnl, summarize,
)


def T(date, code, side, qty, price, fee=0.0):
    return {"날짜": date, "종목코드": code, "구분": side, "수량": qty, "단가": price, "수수료세금": fee}


def test_realized_pnl_uses_average_cost():
    trades = [T("2024-01-02", "005930", "매수", 10, 100), T("2024-01-03", "005930", "매수", 10, 120),
              T("2024-01-04", "005930", "매도", 5, 130)]
    r = realized_pnl(trades)
    assert r["평균단가"].iloc[0] == pytest.approx(110)
    assert r["실현손익(원)"].iloc[0] == pytest.approx((130 - 110) * 5)
    assert r["수익률"].iloc[0] == pytest.approx(20 / 110)


def test_fees_reduce_realized_pnl_and_raise_average_cost():
    trades = [T("2024-01-02", "000660", "매수", 10, 100, fee=100), T("2024-01-05", "000660", "매도", 10, 110, fee=50)]
    r = realized_pnl(trades)
    assert r["평균단가"].iloc[0] == pytest.approx(110)  # (1000 + 100) / 10
    assert r["실현손익(원)"].iloc[0] == pytest.approx((110 - 110) * 10 - 50)


def test_selling_more_than_held_is_rejected():
    with pytest.raises(JournalError):
        realized_pnl([T("2024-01-02", "005930", "매수", 5, 100), T("2024-01-03", "005930", "매도", 6, 100)])
    with pytest.raises(JournalError):
        realized_pnl([T("2024-01-02", "005930", "구매", 5, 100)])
    with pytest.raises(JournalError):
        realized_pnl([T("2024-01-02", "005930", "매수", 0, 100)])


def test_apply_trade_to_holdings_buy_sell_and_delete():
    h = []
    h = apply_trade_to_holdings(h, T("2024-01-02", "5930", "매수", 10, 100))
    assert h == [{"종목코드": "005930", "수량": 10.0, "매입단가": 100.0}]
    h = apply_trade_to_holdings(h, T("2024-01-03", "005930", "매수", 10, 120))
    assert h[0]["수량"] == 20 and h[0]["매입단가"] == pytest.approx(110)
    h = apply_trade_to_holdings(h, T("2024-01-04", "005930", "매도", 5, 130))
    assert h[0]["수량"] == 15 and h[0]["매입단가"] == pytest.approx(110)  # 매도해도 평균단가는 그대로
    h = apply_trade_to_holdings(h, T("2024-01-05", "005930", "매도", 15, 130))
    assert h == []  # 전량 매도하면 목록에서 사라진다
    with pytest.raises(JournalError):
        apply_trade_to_holdings([], T("2024-01-06", "005930", "매도", 1, 100))


def _closes():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    return {"000001": pd.Series([100.0, 110.0, 99.0, 108.9], index=idx)}


def test_daily_return_flows_and_compounding():
    trades = [T("2024-01-02", "000001", "매수", 10, 100)]
    perf = daily_performance(trades, _closes())
    # 첫날: 종가 100에 100에 샀으니 0%
    assert perf["일수익률"].iloc[0] == pytest.approx(0.0)
    assert perf["일수익률"].iloc[1] == pytest.approx(0.10)       # 100 -> 110
    assert perf["일수익률"].iloc[2] == pytest.approx(-0.10)      # 110 -> 99
    assert perf["일수익률"].iloc[3] == pytest.approx(0.10)       # 99 -> 108.9
    assert perf["누적수익률"].iloc[-1] == pytest.approx(1.1 * 0.9 * 1.1 - 1)
    assert perf["일손익(원)"].sum() == pytest.approx(10 * (108.9 - 100))


def test_additional_buy_does_not_count_as_return():
    # 둘째 날 110에 10주 추가 매수 — 넣은 돈은 수익이 아니다
    trades = [T("2024-01-02", "000001", "매수", 10, 100), T("2024-01-03", "000001", "매수", 10, 110)]
    perf = daily_performance(trades, _closes())
    # 둘째 날: 기존 10주는 100->110으로 +100원, 새로 산 10주는 산 가격==종가라 0원. 분모는 어제 평가액+오늘 매수금액
    assert perf["일수익률"].iloc[1] == pytest.approx(100 / (1000 + 1100))
    # 셋째 날: 20주 x (99-110)
    assert perf["일수익률"].iloc[2] == pytest.approx(-0.10)
    assert perf["일손익(원)"].sum() == pytest.approx(10 * (108.9 - 100) + 10 * (108.9 - 110))


def test_sell_day_uses_trade_price_and_total_pnl_matches_realized_plus_unrealized():
    trades = [T("2024-01-02", "000001", "매수", 10, 100, fee=10), T("2024-01-04", "000001", "매도", 4, 105, fee=5)]
    perf = daily_performance(trades, _closes())
    s = summarize(trades, perf)
    assert s["total_pnl"] == pytest.approx(6 * 108.9 + 4 * 105 - 5 - (10 * 100 + 10))
    assert s["realized"] == pytest.approx((105 - 101) * 4 - 5)
    assert s["realized"] + s["unrealized"] == pytest.approx(s["total_pnl"])
    # 마지막 날 남은 6주를 종가(108.9)에 전량 매도하면 평가액은 0이 되고, 그날 수익률은 99->108.9의 +10%
    done = daily_performance(trades + [T("2024-01-05", "000001", "매도", 6, 108.9)], _closes())
    assert done["평가금액"].iloc[-1] == 0 and done["일수익률"].iloc[-1] == pytest.approx(0.10)


def test_missing_prices_or_oversell_raise():
    with pytest.raises(JournalError):
        daily_performance([T("2024-01-02", "ZZZ", "매수", 1, 100)], _closes())
    with pytest.raises(JournalError):
        daily_performance([T("2024-01-02", "000001", "매수", 1, 100), T("2024-01-03", "000001", "매도", 2, 100)], _closes())


def test_trades_storage_roundtrip_local_file(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_TRADES_PATH", tmp_path / "trades.json")
    monkeypatch.setattr(fetch, "_trades_cache", None)
    monkeypatch.setattr(fetch, "_gist_read_file", lambda filename: None)
    monkeypatch.setattr(fetch, "_gist_write_file", lambda filename, content: False)
    assert fetch.load_trades() == []
    trades = [T("2024-01-02", "005930", "매수", 10, 70000)]
    fetch.save_trades(trades)
    assert json.loads((tmp_path / "trades.json").read_text(encoding="utf-8")) == trades
    fetch._trades_cache = None
    assert fetch.load_trades() == trades
