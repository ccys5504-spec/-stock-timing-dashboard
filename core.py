"""app.py의 여러 뷰(탭)가 공통으로 쓰는 설정값/헬퍼 함수 모음.

2026-09-15 정리: 예전에는 app.py 하나에 탭 5개 내용이 전부 들어있어서
800줄 가까이 됐다. 탭별 화면은 views/ 아래로 나누고, 그 탭들이 다 같이
쓰는 "데이터 불러오기+가공", "사이드바 설정값 묶음" 같은 부분만 여기 남겼다.
"""
from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from data.fetch import fetch_ohlcv, get_universe
from screener.scan import scan_signals
from signals import indicators as ind
from strategy.live_portfolio import build_target_portfolio, current_regime, dropped_holdings

logger = logging.getLogger(__name__)

PERIOD_OPTIONS = {"1년": 1, "3년": 3, "5년": 5}

# 탭 이름을 이렇게 상수로 따로 빼둔 이유: 예전엔 app.py/screener.py가
# VIEWS[0], VIEWS[1]처럼 "몇 번째 탭인지"로 서로를 가리켰는데, 2026-09-18에
# 탭 순서를 바꿔달라는 요청을 받고 보니 그 방식은 순서를 바꿀 때마다
# 엉뚱한 탭으로 안내하는 버그가 생기기 쉬웠다(예: "종목 추천에서 클릭하면
# 단일 종목 분석으로 이동"하던 게 다른 탭으로 잘못 이동). 이제는 이름으로
# 가리키므로 VIEWS의 순서를 바꿔도(= 이 리스트의 나열 순서만) 안전하다.
VIEW_HOLDINGS = "💼 내 보유종목"
VIEW_JOURNAL = "📒 매매 기록·수익률"
VIEW_SINGLE_STOCK = "🔍 단일 종목 분석"
VIEW_SCREENER = "🧭 종목 추천(스크리너)"
VIEW_PORTFOLIO_V2 = "🧺 포트폴리오(v2·실험)"
VIEW_OPS_NOTES = "📋 운영 노트"
VIEWS = [VIEW_HOLDINGS, VIEW_JOURNAL, VIEW_SINGLE_STOCK, VIEW_SCREENER, VIEW_PORTFOLIO_V2, VIEW_OPS_NOTES]

CHART_TIMEFRAMES = {"일봉": None, "주봉": "W", "월봉": "ME", "년봉": "YE"}

# 2026-09-18: 월봉/년봉은 캔들 개수가 너무 적어 보이는 문제가 있었다 — 사이드바
# "조회 기간"(예: 3년)만큼만 데이터를 가져오면 년봉은 캔들이 3~4개뿐이라
# 그래프라고 부르기 민망한 수준이었다. 그래서 월봉/년봉을 볼 때는 신호/
# 백테스트와 무관하게 차트 표시용으로만 더 오래된 데이터까지 따로 가져온다.
CHART_EXTRA_YEARS = {"ME": 10, "YE": 25}


@dataclass
class Settings:
    """사이드바에서 사용자가 고른 설정값 묶음. 각 뷰 함수에 그대로 넘겨준다."""

    threshold: int
    volume_filter_mode: bool | str
    volume_multiplier: float
    stop_loss_pct: float | None
    trailing_stop_pct: float | None
    exit_on_signal: bool
    adaptive_exit: bool


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(code: str, years: int) -> pd.DataFrame:
    return fetch_ohlcv(code, years=years)


def prepare(code: str, years: int, settings: Settings) -> pd.DataFrame | None:
    """데이터 로드 + 지표/신호 계산 + 기간 컷까지 한 번에."""
    try:
        raw = load_data(code, years)
    except Exception as e:  # noqa: BLE001 — 화면에는 "데이터 없음"으로 보이지만 원인은 로그에 남긴다
        logger.warning("데이터 조회 실패 %s(%s년): %s: %s", code, years, type(e).__name__, e)
        return None
    df = ind.build_signals(
        raw,
        buy_threshold=settings.threshold,
        sell_threshold=-settings.threshold,
        volume_filter=settings.volume_filter_mode,
        volume_multiplier=settings.volume_multiplier,
    )
    df = df.dropna(subset=["MA_LONG"])
    cutoff = df.index.max() - pd.Timedelta(days=int(years * 365.25))
    df = df[df.index >= cutoff]
    return df if not df.empty else None


@st.cache_data(ttl=1800, show_spinner=False)
def run_screener(markets, top_n, threshold_, vfilter, vmult):
    universe = get_universe(markets=tuple(markets), top_n=top_n)
    return scan_signals(
        universe, years=1,
        buy_threshold=threshold_, sell_threshold=-threshold_,
        volume_filter=vfilter, volume_multiplier=vmult,
    )


V2_UNIVERSE_TOP_N = 100  # 백테스트와 같은 규칙: 그 시점 시가총액 상위 100개가 후보


@st.cache_data(ttl=1800, show_spinner=False)
def run_v2_portfolio(held_codes: tuple[str, ...]) -> dict:
    """v2 전략의 지금 시점 목표 포트폴리오. 시세 조회에 실패한 종목은 개수를 함께 돌려준다."""
    universe = get_universe(markets=("KOSPI", "KOSDAQ"), top_n=V2_UNIVERSE_TOP_N)
    names = dict(zip(universe["Code"], universe["Name"]))
    markets = dict(zip(universe["Code"], universe["Market"]))
    price_data: dict[str, pd.DataFrame] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as ex:
        futures = {ex.submit(fetch_ohlcv, c, 2): c for c in universe["Code"]}
        for fut in as_completed(futures):
            code = futures[fut]
            try:
                price_data[code] = fut.result()
            except Exception as e:  # noqa: BLE001 — 한 종목 실패로 전체를 막지 않되 개수는 보고
                logger.warning("v2 시세 조회 실패 %s: %s", code, e)
                failed.append(code)
    index_data = {}
    for market, code in (("KOSPI", "KS11"), ("KOSDAQ", "KQ11")):
        try:
            index_data[market] = fetch_ohlcv(code, 2)
        except Exception as e:  # noqa: BLE001 — 지수를 못 받으면 배율 1.0(중립)으로 두고 화면에 알린다
            logger.warning("v2 지수 조회 실패 %s: %s", code, e)
    regime = current_regime(index_data)
    held = set(held_codes)
    marcap = dict(zip(universe["Code"], universe["Marcap"]))
    table, as_of = build_target_portfolio(price_data, names, markets, held_codes=held, regime=regime, marcap=marcap)
    return {
        "table": table, "as_of": as_of, "regime": regime, "index_missing": [m for m in ("KOSPI", "KOSDAQ") if m not in index_data], "requested": len(universe), "loaded": len(price_data),
        "failed": [(c, names.get(c, c)) for c in sorted(failed)],
        "dropped": dropped_holdings(table, held, names),
    }


def resample_for_chart(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """일봉 df를 주봉/월봉/년봉으로 리샘플링하고, 그 기준으로 지표를 다시 계산한다
    (차트 표시 전용 — 매매 신호/백테스트는 항상 일봉 기준 그대로 유지된다).
    """
    if rule is None:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    resampled = df[list(agg.keys())].resample(rule).agg(agg).dropna(subset=["Close"])
    return ind.add_indicators(resampled)


def chart_dataframe(code: str, rule: str | None, base_df: pd.DataFrame, years: int) -> pd.DataFrame:
    """차트에 쓸 데이터프레임을 만든다.

    일봉/주봉은 사이드바 조회 기간(base_df)으로 충분하지만, 월봉/년봉은
    캔들이 몇 개 안 남아서 CHART_EXTRA_YEARS만큼 더 오래된 데이터를 따로
    받아와 리샘플링한다(차트 표시 전용 — 신호/백테스트에는 영향 없음).
    추가 조회가 실패하면 원래 base_df로 조용히 되돌아간다.
    """
    extra_years = CHART_EXTRA_YEARS.get(rule)
    if extra_years and extra_years > years:
        try:
            extended_raw = load_data(code, extra_years)
        except Exception as e:  # noqa: BLE001 — 확장 조회가 실패하면 기존 데이터로 그리되 원인은 남긴다
            logger.warning("차트용 확장 조회 실패 %s(%s년): %s: %s", code, extra_years, type(e).__name__, e)
            extended_raw = None
        if extended_raw is not None and not extended_raw.empty:
            return resample_for_chart(extended_raw, rule)
    return resample_for_chart(base_df, rule)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM 포함 -> 엑셀에서 한글 깨짐 방지


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="스캔결과")
    return buf.getvalue()
