"""포트폴리오(v2) 검증 스크립트들이 공통으로 쓰는 데이터 로더.

세 스크립트(portfolio_backtest / portfolio_regime_check / portfolio_cost_stress)가
똑같은 병렬 조회 코드를 각자 갖고 있었고, 조회에 실패한 종목은 `except: continue`로
조용히 빠졌다. 그러면 "유니버스 100개"라고 적고 실제로는 98개로 돌린 결과가
그대로 문서에 들어갈 수 있다. 여기서는 실패한 종목과 사유를 반드시 모아서
출력한다.

주의: 실패 종목이 빠진 채로 결과를 내는 것 자체는 막지 않는다(막으면 종목 하나
때문에 전체 검증이 안 돌아간다). 대신 실패가 있으면 결과 위에 눈에 띄게 표시하고,
결과 요약에 "실제 사용 종목 수"를 함께 적는다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from data.fetch import fetch_ohlcv_range

WARMUP_DAYS = 450  # 13개월 모멘텀 룩백 + 200일 장세 이평선을 모두 커버
MAX_WORKERS = 12


def load_universe_prices(
    universe: pd.DataFrame, fetch_start: str, end: str, warmup_days: int = WARMUP_DAYS,
    max_workers: int = MAX_WORKERS,
) -> tuple[dict[str, pd.DataFrame], dict[str, str], list[tuple[str, str, str]]]:
    """universe(Code/Name/Market)의 각 종목 시세를 병렬로 받아온다.

    반환: (price_data, market_by_code, failed). failed는 (코드, 종목명, 사유) 목록.
    """
    price_data: dict[str, pd.DataFrame] = {}
    market_by_code: dict[str, str] = {}
    failed: list[tuple[str, str, str]] = []

    def _fetch(code: str) -> pd.DataFrame:
        return fetch_ohlcv_range(code, fetch_start, end, warmup_days=warmup_days)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch, row["Code"]): (row["Code"], row["Name"], row["Market"])
            for _, row in universe.iterrows()
        }
        for future in as_completed(futures):
            code, name, market = futures[future]
            try:
                price_data[code] = future.result()
                market_by_code[code] = market
            except Exception as e:  # noqa: BLE001 — 종목 하나 실패로 전체를 멈추진 않되, 반드시 기록한다
                failed.append((code, name, f"{type(e).__name__}: {e}"))
    return price_data, market_by_code, sorted(failed)


def report_load(requested: int, price_data: dict, failed: list[tuple[str, str, str]]) -> None:
    """종목 조회 결과를 출력한다. 실패가 있으면 눈에 띄게 경고한다."""
    print(f"종목 데이터: 요청 {requested}개 → 실제 사용 {len(price_data)}개, 조회 실패 {len(failed)}개")
    if failed:
        print(f"⚠️ 아래 {len(failed)}개 종목은 조회에 실패해서 이 결과에 포함되지 않았습니다:")
        for code, name, reason in failed:
            print(f"   - {name}({code}): {reason}")
    print()
