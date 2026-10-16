"""'매도 완료(삭제)' 버튼을 Streamlit AppTest로 실제 클릭해서 검증한다.
저장소는 메모리 가짜로 바꿔서 진짜 Gist/보유종목 파일을 절대 건드리지 않는다."""
import pytest
from streamlit.testing.v1 import AppTest

import views.holdings as holdings_view


class Store:
    def __init__(self, holdings=None):
        self.holdings = list(holdings or [])


@pytest.fixture
def store(monkeypatch):
    s = Store(holdings=[{"종목코드": "005930", "수량": 10, "매입단가": 70000},
                        {"종목코드": "000660", "수량": 2, "매입단가": 200000}])
    monkeypatch.setattr(holdings_view, "load_holdings", lambda: [dict(h) for h in s.holdings])
    monkeypatch.setattr(holdings_view, "save_holdings", lambda h: setattr(s, "holdings", [dict(x) for x in h]))
    monkeypatch.setattr(holdings_view, "resolve_stock_name", lambda code: f"종목{code}")
    # 현재가 참고값 — 네트워크 조회 없이 고정값으로 (72,000원, 첫 종목 매입가 70,000원보다 위)
    monkeypatch.setattr(holdings_view, "_current_price", lambda code: 72000.0)
    return s


def _sell_app():
    import views.holdings as hv
    hv._render_sell_complete([{"종목코드": "005930", "수량": 10, "매입단가": 70000},
                              {"종목코드": "000660", "수량": 2, "매입단가": 200000}])


def _run():
    return AppTest.from_function(_sell_app, default_timeout=30).run()


def test_row_shows_current_price_and_gain_but_no_sell_price_input(store):
    at = _run()
    assert not at.exception
    text = "".join(m.value for m in at.markdown)
    assert "현재가 72,000원" in text and "+2.9%" in text  # 72,000 / 70,000 - 1
    assert len(at.number_input) == 0  # 매도가 입력칸은 없다(매매 기록 탭을 없앴으므로)


def test_first_click_only_asks_for_confirmation(store):
    at = _run()
    at.button(key="sellbtn_005930").click().run()
    assert not at.exception
    assert at.warning  # 확인 경고 문구가 표시됨
    assert len(store.holdings) == 2  # 아직 아무것도 지워지지 않았다


def test_confirm_deletes_only_that_holding(store):
    at = _run()
    at.button(key="sellbtn_005930").click().run()
    at.button(key="sellconfirm_005930").click().run()
    assert not at.exception
    assert [h["종목코드"] for h in store.holdings] == ["000660"]  # 다른 종목은 그대로


def test_cancel_does_not_delete(store):
    at = _run()
    at.button(key="sellbtn_005930").click().run()
    at.button(key="sellcancel_005930").click().run()
    assert not at.exception
    assert len(store.holdings) == 2
    assert at.button(key="sellbtn_005930")  # 원래 버튼 행으로 돌아온다
