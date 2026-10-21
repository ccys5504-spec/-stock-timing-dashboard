"""시황·전망 화면(views/market_outlook.py)이 쓰는 외부 자료 수집.

  1) 뉴스 헤드라인 — Google 뉴스 RSS(누구나 쓸 수 있는 공개 피드, 인증키 불필요). 제목·언론사·시각·원문 링크만
     가져온다(기사 본문은 저작권이 있으니 저장·재배포하지 않고, 링크로 원문 사이트에 보낸다).
  2) 주요 시장 지표 — FinanceDataReader(코스피·코스닥·환율·미국 지수·VIX·미국 금리·유가·금).

외부에서 받은 글(제목 등)은 신뢰할 수 없는 입력으로 취급한다: HTML 태그를 지우고, 링크는 http/https만 허용하며,
XML에 DTD/ENTITY 선언이 있으면 파싱하지 않는다(엔티티 폭탄 방지).
"""
from __future__ import annotations

import datetime as dt
import html
import logging
import re
import urllib.parse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from email.utils import parsedate_to_datetime

import FinanceDataReader as fdr
import requests

logger = logging.getLogger(__name__)

KST = dt.timezone(dt.timedelta(hours=9))
NEWS_URL = "https://news.google.com/rss/search"
_UA = {"User-Agent": "Mozilla/5.0 (stock-timing-dashboard)"}
MAX_TITLE_CHARS = 150

_TAG_RE = re.compile(r"<[^>]*>")
_SPACE_RE = re.compile(r"\s+")
_KEY_RE = re.compile(r"[^0-9a-z가-힣]")


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    link: str
    published: dt.datetime | None  # KST 기준(알 수 없으면 None)


def _clean_text(raw: str | None) -> str:
    text = html.unescape(_TAG_RE.sub("", raw or ""))
    return _SPACE_RE.sub(" ", text).strip()


def parse_news_rss(content: bytes, limit: int = 15) -> list[NewsItem]:
    """RSS(XML)에서 뉴스 목록을 뽑는다. 최신순, 같은 제목(통신사 기사 재게시 등)은 하나만 남긴다."""
    lowered = content.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise ValueError("RSS에 DTD/ENTITY 선언이 있어 처리하지 않습니다.")
    root = ET.fromstring(content)

    parsed: list[tuple[str, NewsItem]] = []
    for node in root.findall("./channel/item"):
        title = _clean_text(node.findtext("title"))
        source = _clean_text(node.findtext("source"))
        link = (node.findtext("link") or "").strip()
        if not title or urllib.parse.urlparse(link).scheme not in ("http", "https"):
            continue
        if source and title.endswith(f" - {source}"):  # Google 뉴스는 제목 끝에 " - 언론사"를 붙인다
            title = title[: -len(f" - {source}")].rstrip()
        key = _KEY_RE.sub("", title.lower())[:40]
        if not key:
            continue
        try:
            published = parsedate_to_datetime(node.findtext("pubDate") or "").astimezone(KST)
        except (TypeError, ValueError):
            published = None
        if len(title) > MAX_TITLE_CHARS:
            title = title[: MAX_TITLE_CHARS - 1].rstrip() + "…"
        parsed.append((key, NewsItem(title=title, source=source, link=link, published=published)))

    # 최신순으로 정렬한 뒤, 같은 제목(통신사 기사 재게시 등)은 가장 최신 하나만 남긴다
    parsed.sort(key=lambda kv: kv[1].published or dt.datetime.min.replace(tzinfo=KST), reverse=True)
    items: list[NewsItem] = []
    seen: set[str] = set()
    for key, item in parsed:
        if key in seen:
            continue
        seen.add(key)
        items.append(item)
    return items[:limit]


def fetch_news(query: str, limit: int = 12, days: int = 3) -> list[NewsItem]:
    """검색어로 최근 days일 안의 뉴스 헤드라인을 가져온다. 일시적 오류에 대비해 한 번 더 시도한다."""
    params = {"q": f"{query} when:{days}d", "hl": "ko", "gl": "KR", "ceid": "KR:ko"}
    last_err: Exception | None = None
    for _attempt in range(2):
        try:
            r = requests.get(NEWS_URL, params=params, timeout=10, headers=_UA)
            r.raise_for_status()
            return parse_news_rss(r.content, limit)
        except Exception as e:  # noqa: BLE001 — 호출부가 화면에 알리도록 마지막 예외를 다시 던진다
            last_err = e
            logger.warning("뉴스 조회 실패(%s): %s: %s", query, type(e).__name__, e)
    raise last_err  # type: ignore[misc]


def fetch_news_topics(topics: dict[str, tuple[str, int]], limit: int = 12) -> dict[str, tuple[list[NewsItem], str | None]]:
    """{주제: (검색어, 최근 며칠)}을 병렬로 가져온다. 주제별로 (뉴스 목록, 오류 문구)를 돌려주므로 한 주제가
    실패해도 나머지는 정상으로 보여줄 수 있다."""

    def _one(item: tuple[str, tuple[str, int]]):
        label, (query, days) = item
        try:
            return label, (fetch_news(query, limit=limit, days=days), None)
        except Exception as e:  # noqa: BLE001
            return label, ([], f"{type(e).__name__}: {e}")

    if not topics:
        return {}
    with ThreadPoolExecutor(max_workers=min(6, len(topics))) as ex:
        return dict(ex.map(_one, topics.items()))


# ---- 주요 시장 지표 ----------------------------------------------------------------------

# (화면 이름, FinanceDataReader 심볼, 종류) — 종류는 화면에서 숫자 모양을 정할 때 쓴다.
SNAPSHOT_SYMBOLS: list[tuple[str, str, str]] = [
    ("코스피", "KS11", "index"),
    ("코스닥", "KQ11", "index"),
    ("원/달러 환율", "USD/KRW", "fx"),
    ("S&P 500", "US500", "index"),
    ("나스닥", "IXIC", "index"),
    ("다우존스", "DJI", "index"),
    ("VIX(공포지수)", "VIX", "vol"),
    ("미국 10년물 금리", "US10YT", "yield"),
    ("WTI 유가", "CL=F", "commodity"),
    ("금 선물", "GC=F", "commodity"),
]


@dataclass(frozen=True)
class Indicator:
    label: str
    symbol: str
    kind: str
    last: float | None = None
    prev: float | None = None
    date: dt.date | None = None
    error: str | None = None

    @property
    def change(self) -> float | None:
        return None if self.last is None or self.prev is None else self.last - self.prev

    @property
    def change_pct(self) -> float | None:
        return None if self.last is None or not self.prev else self.last / self.prev - 1


def fetch_indicator(label: str, symbol: str, kind: str) -> Indicator:
    """심볼의 최근 종가 두 개(마지막과 그 전)로 값과 전일 대비를 만든다. 실패해도 예외 대신 error를 담아 돌려준다."""
    try:
        start = (dt.date.today() - dt.timedelta(days=21)).isoformat()
        close = fdr.DataReader(symbol, start)["Close"].dropna()
        if len(close) < 2:
            raise ValueError("최근 시세가 2건 미만입니다.")
        return Indicator(label, symbol, kind, last=float(close.iloc[-1]), prev=float(close.iloc[-2]),
                         date=close.index[-1].date())
    except Exception as e:  # noqa: BLE001
        logger.warning("시장 지표 조회 실패(%s): %s: %s", symbol, type(e).__name__, e)
        return Indicator(label, symbol, kind, error=f"{type(e).__name__}: {e}")


def fetch_snapshot() -> list[Indicator]:
    with ThreadPoolExecutor(max_workers=5) as ex:
        return list(ex.map(lambda t: fetch_indicator(*t), SNAPSHOT_SYMBOLS))
