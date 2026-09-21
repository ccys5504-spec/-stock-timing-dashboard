"""매매 기록 화면과 '매도 완료(삭제)' 버튼을 Streamlit AppTest로 실제 클릭해서 검증한다.
저장소는 메모리 가짜로 바꿔서 진짜 Gist/보유종목 파일을 절대 건드리지 않는다."""
import datetime as dt

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import views.holdings as holdings_view
import views.journal as journal_view


class Store:
    def __init__(self, holdings=None, trades=None):
        self.holdings = list(holdings or [])
        self.trades = list(trades or [])


@pytest.fixture
def store(monkeypatch):
    s = Store(
        holdings=[{"종목코드": "005930", "수량": 10, "매입단가": 70000}],
    )
    for mod in (holdings_view, journal_view):
        monkeypatch.setattr(mod, "load_holdings", lambda: [dict(h) for h in s.holdings], raising=False)
        monkeypatch.setattr(mod, "save_holdings", lambda h: setattr(s, "holdings", [dict(x) for x in h]), raising=False)
        monkeypatch.setattr(mod, "load_trades", lambda: [dict(t) for t in s.trades], raising=False)
        monkeypatch.setattr(mod, "save_trades", lambda t: setattr(s, "trades", [dict(x) for x in t]), raising=False)
        monkeypatch.setattr(mod, "resolve_stock_name", lambda code: f"종목{code}", raising=False)
    idx = pd.bdate_range(dt.date.today() - dt.timedelta(days=20), dt.date.today())
    monkeypatch.setattr(journal_view, "_closes", lambda code, start: pd.Series(
        [70000 + 100 * i for i in range(len(idx))], index=idx, dtype=float))
    return s


def _sell_app():
    import views.holdings as hv
    hv._render_sell_complete([{"종목코드": "005930", "수량": 10, "매입단가": 70000}])


def _journal_app():
    import views.journal as jv
    jv.render()


def test_sell_complete_with_price_deletes_holding_and_records_trades(store):
    at = AppTest.from_function(_sell_app).run()
    assert not at.exception
    at.number_input(key="sellpx_005930").set_value(75000).run()
    at.button(key="sellbtn_005930").click().run()
    assert not at.exception
    assert store.holdings == []  # 보유종목에서 삭제·저장됨
    sides = [(t["구분"], t["수량"], t["단가"]) for t in store.trades]
    # 매수 이력이 없어 보유종목의 매입단가로 기초 매수가 함께 남고, 이어서 전량 매도가 남는다
    assert sides == [("매수", 10, 70000), ("매도", 10, 75000)]


def test_sell_complete_without_price_only_deletes(store):
    at = AppTest.from_function(_sell_app).run()
    at.button(key="sellbtn_005930").click().run()
    assert store.holdings == [] and store.trades == []


def test_journal_form_records_trade_and_syncs_holdings(store):
    at = AppTest.from_function(_journal_app).run()
    assert not at.exception
    at.text_input[0].set_value("000660")
    at.number_input[0].set_value(5)
    at.number_input[1].set_value(200000)
    at.button[0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t["종목코드"] for t in store.trades] == ["000660"]
    assert {h["종목코드"] for h in store.holdings} == {"005930", "000660"}  # 매수는 보유종목에도 추가

    # 화면에 결과(일일 수익률 지표)가 그려진다
    assert any("누적 수익률" in m.label for m in at.metric)


def test_journal_form_sell_of_preexisting_holding_adds_opening_buy(store):
    at = AppTest.from_function(_journal_app).run()
    at.text_input[0].set_value("005930")
    at.radio[0].set_value("매도")
    at.number_input[0].set_value(3)
    at.number_input[1].set_value(80000)
    at.button[0].click().run()
    assert not at.exception
    # 기록에 매수 이력이 없으면 보유종목의 매입단가로 기초 매수(3주)가 함께 남는다
    assert [(t["구분"], t["수량"], t["단가"]) for t in store.trades] == [("매수", 3, 70000), ("매도", 3, 80000)]
    assert store.holdings[0]["수량"] == 7  # 보유종목 표도 10 -> 7주로 줄어든다


def test_journal_form_rejects_selling_more_than_held_anywhere(store):
    at = AppTest.from_function(_journal_app).run()
    at.text_input[0].set_value("000660")  # 보유종목에도 기록에도 없는 종목
    at.radio[0].set_value("매도")
    at.number_input[0].set_value(3)
    at.number_input[1].set_value(80000)
    at.button[0].click().run()
    assert store.trades == []
    assert at.error
