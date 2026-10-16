"""코스피 지수 탭: 계산 함수와 화면(AppTest)을 가짜 시세로 검증한다(네트워크 사용 안 함)."""
import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import views.market_index as mv
from tests.helpers import make_ohlcv


def _df(n=700, seed=3, start_price=2500.0):
    dates = pd.bdate_range(end="2026-10-14", periods=n)
    return make_ohlcv(dates, seed=seed, drift=0.0006, vol=0.01, start_price=start_price)


def test_summarize_index_matches_manual_calculation():
    df = _df()
    s = mv.summarize_index(df, days=365)
    close = df["Close"]
    assert s["last"] == pytest.approx(close.iloc[-1])
    assert s["day_change"] == pytest.approx(close.iloc[-1] / close.iloc[-2] - 1)
    window = close[close.index >= close.index[-1] - pd.Timedelta(days=365)]
    assert s["period_return"] == pytest.approx(window.iloc[-1] / window.iloc[0] - 1)
    assert s["high"] == pytest.approx(window.max()) and s["low"] == pytest.approx(window.min())
    assert s["vs_ma200"] == pytest.approx(close.iloc[-1] / close.rolling(200).mean().iloc[-1] - 1)


def test_summarize_index_handles_short_history():
    df = _df(n=50)  # 200일선을 계산할 만큼의 데이터가 없다
    s = mv.summarize_index(df, days=31)
    assert s["vs_ma200"] is None and s["period_return"] is not None


def test_figure_contains_selected_moving_averages_and_respects_chart_type():
    df = _df()
    line = mv._build_figure(df, 365, ["20일선", "200일선"], candle=False)
    assert [t.name for t in line.data] == ["코스피", "20일선", "200일선"]
    assert line.data[0].type == "scatter"
    candle = mv._build_figure(df, 365, [], candle=True)
    assert [t.type for t in candle.data] == ["candlestick"]
    # 기간 시작 지점에도 이동평균 값이 있어야 한다(앞부분 데이터로 미리 계산)
    ma = mv._build_figure(df, 92, ["200일선"], candle=False).data[1]
    assert not np.isnan(np.asarray(ma.y, dtype=float)).any()


@pytest.fixture
def fake_index(monkeypatch):
    calls = []

    def fake_load(code, years):
        calls.append((code, years))
        return _df()

    monkeypatch.setattr(mv, "load_data", fake_load)
    return calls


def _app():
    import views.market_index as v
    v.render()


def test_view_renders_metrics_and_chart_for_default_period(fake_index):
    at = AppTest.from_function(_app, default_timeout=30).run()
    assert not at.exception
    assert fake_index[0][0] == "KS11"
    labels = [m.label for m in at.metric]
    assert any("코스피" in l and "기준" in l for l in labels)
    assert "200일선 대비" in labels and "이 앱의 장세 판단" in labels
    assert any("수익률" in l for l in labels)
    assert at.radio(key="idx_period").value == "1년"  # 기본 기간


def test_changing_period_requests_enough_history_and_still_renders(fake_index):
    at = AppTest.from_function(_app, default_timeout=30).run()
    at.radio(key="idx_period").set_value("10년").run()
    assert not at.exception
    assert fake_index[-1] == ("KS11", 10)  # 10년치(3652일)를 덮는 연수. 이동평균용 앞부분은 load_data가 따로 더 받아온다
    assert any("최근 10년 수익률" in m.label for m in at.metric)
    at.radio(key="idx_chart").set_value("캔들").run()
    assert not at.exception


def test_fetch_failure_is_reported_not_crashing(monkeypatch):
    def boom(code, years):
        raise ConnectionError("시세 서버 응답 없음")

    monkeypatch.setattr(mv, "load_data", boom)
    at = AppTest.from_function(_app, default_timeout=30).run()
    assert not at.exception
    assert any("불러오지 못했습니다" in e.value and "ConnectionError" in e.value for e in at.error)
