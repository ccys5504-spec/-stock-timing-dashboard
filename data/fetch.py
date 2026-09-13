"""국내 주식 일봉 시세 수집 모듈."""
from __future__ import annotations

import datetime as dt
import functools
import json
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd

_WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"
_HOLDINGS_PATH = Path(__file__).parent / "holdings.json"

_DEFAULT_WATCHLIST: dict[str, str] = {
    "SK하이닉스": "000660",
    "두산에너빌리티": "034020",
    "주성엔지니어링": "036930",
}


def load_watchlist() -> dict[str, str]:
    """관심 종목(종목명 -> 종목코드)을 data/watchlist.json에서 읽어온다.

    파일이 없으면 기본값으로 새로 만든다.
    """
    if not _WATCHLIST_PATH.exists():
        save_watchlist(_DEFAULT_WATCHLIST)
        return dict(_DEFAULT_WATCHLIST)
    with open(_WATCHLIST_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_watchlist(watchlist: dict[str, str]) -> None:
    """관심 종목을 data/watchlist.json에 저장한다."""
    with open(_WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(watchlist, f, ensure_ascii=False, indent=2)


# 관심 종목 (종목명 -> 종목코드). 앱에서 교체하면 data/watchlist.json에 저장되고,
# 다음 실행부터 반영된다. 이번 세션 안에서 즉시 반영하려면 load_watchlist()를
# 다시 호출해야 한다(app.py에서 그렇게 처리함).
WATCHLIST: dict[str, str] = load_watchlist()


def load_holdings() -> list[dict]:
    """내 보유종목 목록을 data/holdings.json에서 읽어온다.

    각 항목: {"종목코드": str, "수량": float, "매입단가": float}
    파일이 없으면 빈 목록을 반환한다 (기본으로 보유종목을 가정하지 않음).
    """
    if not _HOLDINGS_PATH.exists():
        return []
    with open(_HOLDINGS_PATH, encoding="utf-8") as f:
        return json.load(f)


def save_holdings(holdings: list[dict]) -> None:
    """내 보유종목 목록을 data/holdings.json에 저장한다."""
    with open(_HOLDINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(holdings, f, ensure_ascii=False, indent=2)


@functools.lru_cache(maxsize=1)
def _full_krx_listing() -> pd.DataFrame:
    """전체 KRX 종목 목록(코드 -> 종목명 조회용)을 한 번만 받아와 캐싱한다."""
    return fdr.StockListing("KRX")


def resolve_stock_name(code: str) -> str | None:
    """종목코드로 종목명을 찾는다. 못 찾거나 조회에 실패하면 None."""
    try:
        listing = _full_krx_listing()
        match = listing[listing["Code"] == code]
    except Exception:  # noqa: BLE001
        return None
    if match.empty:
        return None
    return str(match.iloc[0]["Name"])


def fetch_ohlcv(code: str, years: int = 3) -> pd.DataFrame:
    """종목코드로 최근 N년치 일봉 OHLCV 데이터를 가져온다.

    Returns columns: Open, High, Low, Close, Volume (index: 날짜)
    """
    end = dt.date.today()
    # 지표 계산 여유분: 상승장/하락장 자동판별용 200일 이동평균이 첫날부터
    # 유효하려면 최소 200 거래일(약 290 달력일) 이상의 워밍업이 필요하다.
    start = end - dt.timedelta(days=int(years * 365.25) + 300)
    return fetch_ohlcv_range(code, start.isoformat(), end.isoformat())


def get_universe(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    top_n: int = 50,
) -> pd.DataFrame:
    """시가총액 상위 N개 종목 목록을 가져온다 (우선주·관리종목·스팩 등은 제외).

    Returns columns: Code, Name, Market, Marcap
    """
    import re

    listing = fdr.StockListing("KRX")
    listing = listing[listing["Market"].isin(markets)].copy()

    # 우선주 제외 (종목명이 '...우', '...우B', '...2우B' 등으로 끝남)
    is_preferred = listing["Name"].str.contains(r"\d?우[A-Z]?$", regex=True)
    listing = listing[~is_preferred]

    # 관리종목/투자주의환기종목/스팩 등 제외
    exclude_keywords = "관리종목|투자주의|SPAC|스팩"
    if "Dept" in listing.columns:
        listing = listing[~listing["Dept"].astype(str).str.contains(exclude_keywords, regex=True, na=False)]

    listing = listing.sort_values("Marcap", ascending=False).head(top_n)
    return listing[["Code", "Name", "Market", "Marcap"]].reset_index(drop=True)


def fetch_ohlcv_range(code: str, start: str, end: str, warmup_days: int = 300) -> pd.DataFrame:
    """종목코드로 [start, end] 구간의 일봉 OHLCV 데이터를 가져온다.

    이평선(MA60), 상승장/하락장 자동판별(200일 이평선) 등 지표 계산에 필요한
    워밍업 구간(기본 300일)만큼 start 이전 데이터도 함께 가져온다 — 신호/백테스트
    코드에서 실제 표시 구간만 다시 잘라서 쓰면 된다.

    start/end: "YYYY-MM-DD" 형식 문자열
    """
    start_date = dt.date.fromisoformat(start) - dt.timedelta(days=warmup_days)
    df = fdr.DataReader(code, start_date.isoformat(), end)
    if df.empty:
        raise ValueError(f"'{code}' 종목의 데이터를 가져오지 못했습니다.")
    df = df.dropna(subset=["Close"]).copy()
    return df
