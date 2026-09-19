"""여러 종목을 한 번에 스캔해서 현재 매수/매도 신호가 뜬 종목을 찾는 스크리너.

⚠️ 여기서 나오는 결과는 기술적 지표를 기계적으로 계산한 것일 뿐, 투자 자문이
아니다. 시가총액 상위 종목 위주로 스캔하는 이유는 (1) 거래량이 많아 신호의
신뢰도가 상대적으로 낫고 (2) 관심 종목 3~4개 수준으로 추리기 좋기 때문이다.

2026-09-20: 예전에는 종목별 조회/계산이 실패하면 조용히 버려서(`except: return
None`) 100개를 요청해도 98개만 스캔됐는지 알 수 없었다. 종목 하나가 실패해도
전체 스캔이 멈추지 않게 하는 건 그대로 두되, 실패한 종목과 사유를 모아서
결과 DataFrame의 `attrs`(`requested`, `failed`)에 담아 화면에서 보여줄 수 있게
했다.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.fetch import fetch_ohlcv
from signals.indicators import build_signals

MAX_WORKERS = 8
logger = logging.getLogger(__name__)


def _scan_one(code: str, name: str, market: str, marcap: float, years: int, signal_kwargs: dict):
    """(결과 행, 실패 정보) 중 정확히 하나만 None이 아니다."""
    try:
        raw = fetch_ohlcv(code, years=years)
        df = build_signals(raw, **signal_kwargs)
        df = df.dropna(subset=["MA_LONG"])
        if df.empty:
            return None, (code, name, "지표를 계산할 만큼 데이터가 없음(상장 직후 등)")
        latest = df.iloc[-1]
        row = {
            "종목명": name,
            "종목코드": code,
            "시장": market,
            "현재가": float(latest["Close"]),
            "신호": latest["SIGNAL"],
            "점수": int(latest["SCORE_TOTAL"]),
            "RSI": round(float(latest["RSI"]), 1),
            "거래량확인": bool(latest["VOLUME_OK"]),
            "시가총액(억)": round(marcap / 1e8) if pd.notna(marcap) else None,
            "기준일": latest.name.date().isoformat(),
        }
        return row, None
    except Exception as e:  # noqa: BLE001 — 한 종목 실패가 전체 스캔을 멈추면 안 되지만 조용히 삼키지는 않는다
        logger.warning("스캔 실패 %s(%s): %s: %s", name, code, type(e).__name__, e)
        return None, (code, name, f"{type(e).__name__}: {e}")


def scan_signals(
    candidates: pd.DataFrame,
    years: int = 1,
    max_workers: int = MAX_WORKERS,
    **signal_kwargs,
) -> pd.DataFrame:
    """candidates(get_universe 결과)의 각 종목에 대해 현재 시점 신호를 계산한다.

    signal_kwargs는 signals.indicators.build_signals로 전달된다
    (buy_threshold, sell_threshold, volume_filter, volume_multiplier).
    네트워크 호출이 많으므로 스레드풀로 병렬 조회한다.

    반환 DataFrame의 `attrs`에 스캔 결과 요약이 들어 있다:
      - requested: 스캔을 요청한 종목 수
      - failed: [(종목코드, 종목명, 사유), ...] 조회/계산에 실패한 종목
    """
    rows = []
    failed: list[tuple[str, str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _scan_one, row["Code"], row["Name"], row["Market"], row["Marcap"], years, signal_kwargs
            ): row["Code"]
            for _, row in candidates.iterrows()
        }
        for future in as_completed(futures):
            row, error = future.result()
            if row is not None:
                rows.append(row)
            if error is not None:
                failed.append(error)

    result_df = pd.DataFrame(rows)
    if not result_df.empty:
        result_df = result_df.sort_values(["점수", "시가총액(억)"], ascending=[False, False]).reset_index(drop=True)
    result_df.attrs["requested"] = len(candidates)
    result_df.attrs["failed"] = sorted(failed)
    return result_df
