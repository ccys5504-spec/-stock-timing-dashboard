"""종목 추천 화면: '오늘의 추천'과 '직접 범위 스캔'을 합친 구조(시총 상위 300개 자동 스캔)를 AppTest로 검증한다.
실제 시세 조회는 하지 않고 run_screener를 가짜로 바꾼다."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import views.screener as sv


def _scan(with_buy=True):
    rows = [
        {"종목명": "가나다", "종목코드": "000001", "시장": "KOSPI", "현재가": 10000.0,
         "신호": "매수" if with_buy else "관망", "점수": 2, "RSI": 45.0, "거래량확인": True,
         "시가총액(억)": 5000, "기준일": "2026-09-28"},
        {"종목명": "라마바", "종목코드": "000002", "시장": "KOSDAQ", "현재가": 20000.0,
         "신호": "매도", "점수": -2, "RSI": 75.0, "거래량확인": False, "시가총액(억)": 3000, "기준일": "2026-09-28"},
        {"종목명": "사아자", "종목코드": "000003", "시장": "KOSPI", "현재가": 30000.0,
         "신호": "관망", "점수": 1, "RSI": 50.0, "거래량확인": False, "시가총액(억)": 2000, "기준일": "2026-09-28"},
    ]
    df = pd.DataFrame(rows)
    df.attrs["requested"] = 4
    df.attrs["failed"] = [("999999", "실패종목", "ConnectionError: x")]
    return df


class FakeScreener:
    def __init__(self, result):
        self.result = result
        self.calls = []
        self.cleared = 0
        self.cleared_at_call = []

    def __call__(self, markets, top_n, threshold, vfilter, vmult):
        self.calls.append((list(markets), top_n, threshold, vfilter, vmult))
        self.cleared_at_call.append(self.cleared)  # 이 호출 시점까지 캐시를 몇 번 비웠는지
        return self.result

    def clear(self):
        self.cleared += 1


@pytest.fixture
def fake(monkeypatch):
    f = FakeScreener(_scan())
    monkeypatch.setattr(sv, "run_screener", f)
    monkeypatch.setattr(sv, "prepare", lambda code, years, settings: None)  # 관심종목 폴백 조회는 하지 않는다
    return f


def _app():
    import views.screener as v
    from core import Settings
    settings = Settings(threshold=2, volume_filter_mode="auto", volume_multiplier=1.4, stop_loss_pct=None,
                        trailing_stop_pct=None, exit_on_signal=True, adaptive_exit=False)
    v.render({"관심종목": "000002"}, 3, settings)


def test_tab_opens_with_one_automatic_300_stock_scan_and_no_manual_scan_controls(fake):
    at = AppTest.from_function(_app, default_timeout=60).run()
    assert not at.exception
    assert fake.calls == [(["KOSPI", "KOSDAQ"], 300, 2, "auto", 1.4)]  # 열자마자 시총 상위 300개, 사이드바 설정 그대로
    assert sv.TODAY_PICK_UNIVERSE == 300
    # 예전 '직접 범위를 정해서 스캔하기' 컨트롤은 없다
    assert len(at.slider) == 0 and len(at.multiselect) == 0
    assert not [b for b in at.button if "스캔 시작" in b.label]
    assert "직접 범위" not in "".join(s.value for s in at.subheader)


def test_pick_card_table_metrics_and_watchlist_section_share_the_same_scan(fake):
    at = AppTest.from_function(_app, default_timeout=60).run()
    text = "".join(m.value for m in at.markdown)
    assert "오늘의 추천" in text and "🟢 가나다 (000001)" in text  # 매수 신호 1위가 추천 카드로
    labels = {m.label: m.value for m in at.metric}
    assert labels["스캔한 종목 수"] == "3" and labels["매수 신호"] == "1개" and labels["매도 신호"] == "1개"
    assert any("매수/매도 신호만 보기" in c.label for c in at.checkbox)
    assert any("관심종목 교체 제안" in s.value for s in at.subheader)
    assert len(fake.calls) == 1  # 추천 카드·표·교체 제안이 한 번의 스캔 결과를 같이 쓴다


def test_no_buy_signal_shows_message_but_still_shows_table(monkeypatch):
    f = FakeScreener(_scan(with_buy=False))
    monkeypatch.setattr(sv, "run_screener", f)
    monkeypatch.setattr(sv, "prepare", lambda code, years, settings: None)
    at = AppTest.from_function(_app, default_timeout=60).run()
    assert not at.exception
    assert any("매수 신호가 뜬 종목이 없습니다" in i.value for i in at.info)
    assert any("스캔 결과" in s.value for s in at.subheader)


def test_rescan_button_clears_cache_and_scans_again(fake):
    at = AppTest.from_function(_app, default_timeout=60).run()
    at.button(key="rescan").click().run()
    assert not at.exception
    assert fake.cleared == 1
    # 클릭한 실행에서는 (캐시가 아직 있는 채로) 한 번 부르고, 캐시를 비운 뒤 다시 실행되면서 새로 스캔한다
    assert fake.cleared_at_call[-1] == 1 and fake.cleared_at_call[0] == 0


def test_scan_failure_is_reported_not_crashing(monkeypatch):
    class Boom(FakeScreener):
        def __call__(self, *a, **k):
            raise RuntimeError("network down")

    b = Boom(None)
    monkeypatch.setattr(sv, "run_screener", b)
    at = AppTest.from_function(_app, default_timeout=60).run()
    assert not at.exception
    assert any("계산하지 못했습니다" in w.value for w in at.warning)
