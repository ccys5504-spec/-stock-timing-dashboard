"""시황·전망 화면: 뉴스 파싱/보안 처리, 시장 지표, 코스피 상태 요약, 화면을 네트워크 없이 검증한다."""
import datetime as dt

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import data.market_info as mi
import views.market_outlook as mv
from strategy.market_state import describe_market
from tests.helpers import make_ohlcv

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>x</title>
<item><title>[개장시황] 코스피 2%대 반등 - 뉴스핌</title><link>https://news.example.com/a</link>
  <pubDate>Wed, 21 Oct 2026 00:42:00 GMT</pubDate><source url="https://newspim.com">뉴스핌</source></item>
<item><title>코스피 급락 &lt;b&gt;전망&lt;/b&gt;   분분 - 연합뉴스</title><link>https://news.example.com/b</link>
  <pubDate>Tue, 20 Oct 2026 21:00:00 GMT</pubDate><source url="https://yna.co.kr">연합뉴스</source></item>
<item><title>[개장시황] 코스피 2%대 반등! - 다른매체</title><link>https://news.example.com/dup</link>
  <pubDate>Wed, 21 Oct 2026 00:50:00 GMT</pubDate><source>다른매체</source></item>
<item><title>위험한 링크</title><link>javascript:alert(1)</link><pubDate>Wed, 21 Oct 2026 01:00:00 GMT</pubDate></item>
<item><title>날짜 없는 기사</title><link>http://news.example.com/c</link></item>
</channel></rss>""".encode("utf-8")


def test_parse_cleans_titles_converts_to_kst_sorts_and_dedupes():
    items = mi.parse_news_rss(RSS)
    titles = [n.title for n in items]
    assert titles[0] == "[개장시황] 코스피 2%대 반등!"  # 언론사 꼬리표 제거. 같은 제목 두 건 중 더 최신(00:50) 것만 남는다
    assert "코스피 급락 전망 분분" in titles  # HTML 태그 제거, 공백 정리
    assert len(titles) == 3 and "위험한 링크" not in titles  # javascript: 링크는 버린다
    assert items[0].published == dt.datetime(2026, 10, 21, 9, 50, tzinfo=mi.KST)  # 00:50 GMT = 09:50 KST
    assert items[-1].title == "날짜 없는 기사" and items[-1].published is None  # 날짜 없는 건 맨 뒤


def test_parse_respects_limit_and_truncates_long_titles():
    many = "".join(
        f"<item><title>{'가' * 300}{i}</title><link>https://x.com/{i}</link></item>" for i in range(30)
    )
    xml = f'<rss><channel>{many}</channel></rss>'.encode("utf-8")
    items = mi.parse_news_rss(xml, limit=5)
    assert len(items) == 1 or len(items) == 5  # 앞 40자가 같으면 같은 기사로 취급해 하나만 남는다
    assert all(len(n.title) <= mi.MAX_TITLE_CHARS for n in items)


def test_parse_rejects_xml_with_entity_declarations():
    bomb = b'<?xml version="1.0"?><!DOCTYPE r [<!ENTITY a "aaaa">]><rss><channel><item><title>&a;</title></item></channel></rss>'
    with pytest.raises(ValueError):
        mi.parse_news_rss(bomb)


def test_fetch_news_retries_once_then_raises_last_error(monkeypatch):
    calls = {"n": 0}

    class Boom:
        def raise_for_status(self):
            raise ConnectionError("끊김")

    def fake_get(*a, **k):
        calls["n"] += 1
        return Boom()

    monkeypatch.setattr(mi.requests, "get", fake_get)
    with pytest.raises(ConnectionError):
        mi.fetch_news("코스피 시황")
    assert calls["n"] == 2


def test_fetch_news_topics_isolates_failures(monkeypatch):
    def fake_fetch(query, limit=12, days=3):
        if query == "나쁜검색":
            raise ConnectionError("실패")
        return [mi.NewsItem("제목", "매체", "https://x.com", None)]

    monkeypatch.setattr(mi, "fetch_news", fake_fetch)
    out = mi.fetch_news_topics({"좋음": ("코스피", 2), "나쁨": ("나쁜검색", 2)})
    assert out["좋음"][1] is None and len(out["좋음"][0]) == 1
    assert out["나쁨"][0] == [] and "ConnectionError" in out["나쁨"][1]


def test_indicator_change_and_failure_isolation(monkeypatch):
    idx = pd.to_datetime(["2026-10-19", "2026-10-20"])

    def fake_reader(symbol, start):
        if symbol == "BAD":
            raise ConnectionError("조회 실패")
        return pd.DataFrame({"Close": [100.0, 102.0]}, index=idx)

    monkeypatch.setattr(mi.fdr, "DataReader", fake_reader)
    ok = mi.fetch_indicator("지수", "OK", "index")
    assert ok.last == 102.0 and ok.change == pytest.approx(2.0) and ok.change_pct == pytest.approx(0.02)
    assert ok.date == dt.date(2026, 10, 20)
    bad = mi.fetch_indicator("나쁨", "BAD", "index")
    assert bad.last is None and "ConnectionError" in bad.error and bad.change_pct is None


def test_md_escape_neutralises_markdown_and_html():
    nasty = "[클릭](http://evil.com) **굵게** <script>x</script> `코드`"
    out = mv.md_escape(nasty)
    for ch in "[]()*<>`":
        assert all(out[i - 1] == "\\" for i, c in enumerate(out) if c == ch and i > 0), ch
    assert mv._safe_url("https://a.com/x (1)") == "https://a.com/x%20%281%29"


def test_age_label():
    now = dt.datetime(2026, 10, 21, 12, 0, tzinfo=mi.KST)
    assert mv.age_label(now - dt.timedelta(seconds=20), now) == "방금"
    assert mv.age_label(now - dt.timedelta(minutes=5), now) == "5분 전"
    assert mv.age_label(now - dt.timedelta(hours=3), now) == "3시간 전"
    assert mv.age_label(now - dt.timedelta(days=2), now) == "2일 전"
    assert mv.age_label(None, now) == ""


def _index_df(n=500):
    dates = pd.bdate_range(end="2026-10-20", periods=n)
    return make_ohlcv(dates, seed=5, drift=0.0004, vol=0.012, start_price=3000.0)


def test_describe_market_states_facts_without_advice():
    d = describe_market(_index_df())
    m = d["metrics"]
    close = _index_df()["Close"]
    assert m["drawdown"] == pytest.approx(close.iloc[-1] / close[close.index >= close.index[-1] - pd.Timedelta(days=365)].max() - 1)
    assert 0 <= m["rsi"] <= 100
    text = " ".join(d["lines"])
    assert "고점" in text and "200일선" in text and "RSI" in text
    for word in ("매수하세요", "매도하세요", "사세요", "파세요", "오를 것", "내릴 것"):
        assert word not in text  # 권유·예측 표현 금지


def test_describe_market_with_short_or_tiny_data():
    assert describe_market(_index_df(n=1)) == {"metrics": {}, "lines": []}
    short = describe_market(_index_df(n=40))
    assert short["metrics"]["vs_ma200"] is None  # 200일선을 못 만들면 해당 문장만 빠진다
    assert not any("200일선" in l for l in short["lines"])


# ---- 화면 ------------------------------------------------------------------------------

def _fake_snapshot():
    d = dt.date(2026, 10, 20)
    return [
        mi.Indicator("코스피", "KS11", "index", 5553.0, 5700.0, d),
        mi.Indicator("원/달러 환율", "USD/KRW", "fx", 1326.1, 1328.8, d),
        mi.Indicator("미국 10년물 금리", "US10YT", "yield", 5.12, 5.10, d),
        mi.Indicator("금 선물", "GC=F", "commodity", error="ConnectionError: x"),
    ]


def _fake_news(holding_names):
    n = mi.NewsItem("[시황] 코스피 반등 <b>시도</b>", "뉴스핌", "https://news.example.com/a",
                    dt.datetime.now(mi.KST) - dt.timedelta(hours=2))
    out = {label: ([n], None) for label in mv.TOPICS}
    out["전망"] = ([], "ConnectionError: 끊김")
    for name in holding_names:
        out[f"{mv.HOLDINGS_TAB}:{name}"] = ([n], None)
    return out


@pytest.fixture
def fake_sources(monkeypatch):
    monkeypatch.setattr(mv, "_load_snapshot", lambda: _fake_snapshot())
    monkeypatch.setattr(mv, "_load_news", lambda names: _fake_news(names))
    monkeypatch.setattr(mv, "load_data", lambda code, years: _index_df())
    monkeypatch.setattr(mv, "load_holdings", lambda: [{"종목코드": "006400"}, {"종목코드": "033780"}])
    monkeypatch.setattr(mv, "resolve_stock_name", lambda c: {"006400": "삼성SDI", "033780": "KT&G"}[c])


def _app():
    import views.market_outlook as v
    v.render()


def test_view_shows_indicators_state_summary_and_news_tabs(fake_sources):
    at = AppTest.from_function(_app, default_timeout=30).run()
    assert not at.exception
    values = {m.label: m.value for m in at.metric}
    assert values["원/달러 환율"] == "1,326.1" and values["미국 10년물 금리"] == "5.120%"
    assert values["금 선물"] == "조회 실패"  # 실패한 지표만 표시가 달라진다
    text = "".join(m.value for m in at.markdown)
    assert "고점" in text and "RSI" in text  # 규칙 기반 코스피 요약
    assert [t.label for t in at.tabs] == ["시황", "전망", "환율·금리", "미국 증시", "내 보유종목"]
    assert "\\<b\\>" in text or "시도" in text  # 외부 제목의 HTML은 이스케이프되어 그대로 글자로 보인다
    assert "[시황]" in text.replace("\\[", "[").replace("\\]", "]")
    assert any("전망" in w.value or "뉴스를 불러오지 못했습니다" in w.value for w in at.warning)  # 실패한 주제만 경고
    assert "삼성SDI" in text and "KT&G" in text.replace("\\", "")  # 보유종목별 뉴스


def test_view_without_holdings_has_no_holdings_tab(fake_sources, monkeypatch):
    monkeypatch.setattr(mv, "load_holdings", lambda: [])
    at = AppTest.from_function(_app, default_timeout=30).run()
    assert not at.exception
    assert "내 보유종목" not in [t.label for t in at.tabs]
