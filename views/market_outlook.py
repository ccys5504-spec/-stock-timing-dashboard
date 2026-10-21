"""뷰: 시황·전망 — 시장 상황과 도움이 될 만한 소식을 자동으로 모아 보여준다.

구성: ① 주요 시장 지표(코스피·환율·미국 증시·금리·유가 등) ② 이 앱이 규칙대로 요약한 코스피 상태
③ 뉴스 헤드라인(시황 / 전망 / 환율·금리 / 미국 증시 / 내 보유종목).

⚠️ "전망"은 언론이 보도한 **다른 사람들의 의견 모음**일 뿐 이 앱의 예측이 아니다. 뉴스는 제목과 원문 링크만
보여주고, 투자 자문이 아니다. 자료 수집 방법은 data/market_info.py 참고.
"""
from __future__ import annotations

import datetime as dt
import re

import streamlit as st

from core import load_data
from data.fetch import load_holdings, resolve_stock_name
from data.market_info import KST, Indicator, NewsItem, fetch_news_topics, fetch_snapshot
from strategy.market_state import describe_market

NEWS_TTL = 900  # 뉴스는 15분마다 새로 받는다
MAX_HOLDING_NEWS = 5  # 보유종목 뉴스는 앞에서 5종목까지

# 화면 이름 -> (검색어, 최근 며칠). 보유종목 뉴스는 아래 _load_news가 종목명으로 따로 만든다.
TOPICS: dict[str, tuple[str, int]] = {
    "시황": ("코스피 시황", 2),
    "전망": ("코스피 전망", 3),
    "환율·금리": ("원달러 환율 금리", 3),
    "미국 증시": ("뉴욕증시 마감", 2),
}
HOLDINGS_TAB = "내 보유종목"

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+!|<>~$])")


def md_escape(text: str) -> str:
    """외부에서 온 글이 마크다운 서식·링크로 해석되지 않도록 특수문자 앞에 역슬래시를 붙인다."""
    return _MD_SPECIAL.sub(r"\\\1", text)


def _safe_url(url: str) -> str:
    """마크다운 링크 주소를 깨뜨릴 수 있는 문자(공백·괄호)를 퍼센트 인코딩한다."""
    return url.replace(" ", "%20").replace("(", "%28").replace(")", "%29")


def age_label(published: dt.datetime | None, now: dt.datetime | None = None) -> str:
    if published is None:
        return ""
    now = now or dt.datetime.now(KST)
    minutes = int((now - published).total_seconds() // 60)
    if minutes < 1:
        return "방금"
    if minutes < 60:
        return f"{minutes}분 전"
    if minutes < 60 * 24:
        return f"{minutes // 60}시간 전"
    return f"{minutes // (60 * 24)}일 전"


@st.cache_data(ttl=NEWS_TTL, show_spinner=False)
def _load_news(holding_names: tuple[str, ...]) -> dict[str, tuple[list[NewsItem], str | None]]:
    topics = dict(TOPICS)
    for name in holding_names:
        topics[f"{HOLDINGS_TAB}:{name}"] = (name, 5)
    return fetch_news_topics(topics)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_snapshot() -> list[Indicator]:
    return fetch_snapshot()


def _fmt_indicator(ind: Indicator) -> tuple[str, str | None]:
    """(표시 값, 전일 대비 문구)."""
    if ind.last is None:
        return "조회 실패", None
    if ind.kind == "yield":
        delta = f"{ind.change:+.3f}%p" if ind.change is not None else None
        return f"{ind.last:.3f}%", delta
    value = f"{ind.last:,.1f}" if ind.kind == "fx" else f"{ind.last:,.2f}"
    pct = ind.change_pct
    return value, (f"{pct:+.2%}" if pct is not None else None)


def _render_snapshot() -> None:
    st.subheader("🌐 주요 시장 지표")
    try:
        indicators = _load_snapshot()
    except Exception as e:  # noqa: BLE001
        st.error(f"시장 지표를 불러오지 못했습니다: {type(e).__name__}: {e}")
        return
    per_row = 4  # 한 줄에 5개는 좁은 화면에서 숫자가 잘려서 4개로 한다
    for start in range(0, len(indicators), per_row):
        cols = st.columns(per_row)
        for col, ind in zip(cols, indicators[start:start + per_row]):
            value, delta = _fmt_indicator(ind)
            col.metric(ind.label, value, delta, help=(ind.error or (f"{ind.date} 기준" if ind.date else None)))
            if ind.date:
                col.caption(f"{ind.date:%m/%d} 기준")
    failed = [i.label for i in indicators if i.last is None]
    if failed:
        st.caption("⚠️ 불러오지 못한 지표: " + ", ".join(failed))
    st.caption(
        "미국 지수·금리는 한국 시각으로 전날 마감 값이고, 유가·금은 선물 가격입니다. 금리는 변동폭을 %p로 표시합니다."
    )


def _render_market_state() -> None:
    st.subheader("🧭 이 앱이 본 코스피 상태")
    try:
        summary = describe_market(load_data("KS11", 2))
    except Exception as e:  # noqa: BLE001
        st.error(f"코스피 지수를 불러오지 못했습니다: {type(e).__name__}: {e}")
        return
    if not summary["lines"]:
        st.info("요약할 데이터가 부족합니다.")
        return
    st.markdown("\n".join(f"- {line}" for line in summary["lines"]))
    st.caption("지난 가격 움직임을 규칙대로 풀어 쓴 요약입니다. 앞날을 예측하거나 매매를 권하는 내용이 아닙니다.")


def _render_news_list(items: list[NewsItem], error: str | None) -> None:
    if error:
        st.warning(f"뉴스를 불러오지 못했습니다({error}). 아래 '🔄 새로고침'을 눌러보세요.")
        return
    if not items:
        st.caption("최근 기사가 없습니다.")
        return
    lines = []
    for n in items:
        meta = " · ".join(x for x in (md_escape(n.source), age_label(n.published)) if x)
        lines.append(f"- [{md_escape(n.title)}]({_safe_url(n.link)})" + (f" — {meta}" if meta else ""))
    st.markdown("\n".join(lines))


def _render_news(holding_names: tuple[str, ...]) -> None:
    st.subheader("📰 뉴스 헤드라인")
    with st.spinner("뉴스를 모으는 중..."):
        news = _load_news(holding_names)
    labels = list(TOPICS) + ([HOLDINGS_TAB] if holding_names else [])
    for label, tab in zip(labels, st.tabs(labels)):
        with tab:
            if label != HOLDINGS_TAB:
                items, err = news[label]
                _render_news_list(items, err)
                continue
            for name in holding_names:
                st.markdown(f"**{md_escape(name)}**")
                items, err = news[f"{HOLDINGS_TAB}:{name}"]
                _render_news_list(items[:5], err)
    st.caption(
        "출처: Google 뉴스 검색 결과의 헤드라인(각 언론사 원문으로 연결). '전망' 탭은 언론이 보도한 **다른 사람들의 의견**"
        "이며 이 앱의 예측이 아닙니다. 내용의 정확성은 원문에서 확인하세요."
    )


def render() -> None:
    st.caption(
        "시장 상황과 도움이 될 만한 소식을 자동으로 모아 보여줍니다. 지표는 30분, 뉴스는 15분마다 새로 받아옵니다. "
        "투자 자문이 아니며, 중요한 판단 전에는 원문과 증권사 화면을 직접 확인하세요."
    )
    if st.button("🔄 새로고침", key="outlook_refresh", help="저장해 둔 지표·뉴스를 버리고 지금 새로 받아옵니다."):
        _load_news.clear()
        _load_snapshot.clear()
        st.rerun()

    _render_snapshot()
    st.divider()
    _render_market_state()
    st.divider()
    codes = [str(h.get("종목코드", "")).strip().zfill(6) for h in load_holdings() if h.get("종목코드")]
    names = tuple(dict.fromkeys(resolve_stock_name(c) or c for c in codes))[:MAX_HOLDING_NEWS]
    _render_news(names)
