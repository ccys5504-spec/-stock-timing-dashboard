"""상대강도(모멘텀) 기반 종목 순위 — PROMPT_V2.md Phase A2.

"언제 살까"(v1, backtest/engine.py)가 아니라 "무엇을 살까"를 정하는 모듈이다.
포트폴리오 엔진(backtest/portfolio_engine.py)이 매 리밸런싱 시점마다 호출한다.

파라미터(룩백 개월수, skip-1-month, 거래대금 하한)는 데이터에 맞춰 튜닝하지
않고 문헌에서 흔히 쓰는 표준값으로 고정한다 — scripts/optimize.py가 겪었던
"조합을 늘릴수록 과최적화 위험이 커지는" 함정을 되풀이하지 않기 위함
(PROMPT_V2.md "검증 방법론" 1번 참고).
"""
from __future__ import annotations

import pandas as pd

# 상대강도 계산에 쓰는 룩백 구간(개월). 최근 SKIP_MONTHS는 단기 반전 효과를
# 줄이기 위해 계산에서 제외한다(skip-1-month, 모멘텀 연구에서 흔한 관행).
LOOKBACK_MONTHS = (3, 6, 12)
SKIP_MONTHS = 1

# 최근 20거래일 평균 거래대금이 이 금액(원) 미만인 종목은 저유동성으로 보고
# 순위 계산 대상에서 제외한다.
MIN_AVG_TRADING_VALUE = 500_000_000
TRADING_VALUE_WINDOW = 20


def _as_of_price(close: pd.Series, on_or_before: pd.Timestamp) -> float | None:
    """on_or_before 시점 이전(포함)의 가장 최근 종가. 없으면 None."""
    sliced = close[close.index <= on_or_before]
    if sliced.empty:
        return None
    return float(sliced.iloc[-1])


def momentum_score(close: pd.Series, as_of: pd.Timestamp) -> float | None:
    """as_of 시점까지의 종가만 사용해서(미래 데이터 사용 금지) 모멘텀 점수를
    계산한다. [as_of - SKIP_MONTHS, as_of] 구간은 제외하고, 그 기준일로부터
    3/6/12개월 전 대비 수익률을 단순평균한다.

    필요한 만큼의 과거 데이터(최대 룩백 + skip)가 없는 종목(상장한 지 얼마 안
    된 종목 등)은 None을 반환한다 — 순위 계산에서 자동으로 빠진다.
    """
    close = close[close.index <= as_of]
    if close.empty:
        return None

    ref_date = as_of - pd.DateOffset(months=SKIP_MONTHS)
    ref_price = _as_of_price(close, ref_date)
    if ref_price is None or ref_price <= 0:
        return None

    returns = []
    for months in LOOKBACK_MONTHS:
        start_date = ref_date - pd.DateOffset(months=months)
        start_price = _as_of_price(close, start_date)
        if start_price is None or start_price <= 0:
            return None
        returns.append(ref_price / start_price - 1)

    return sum(returns) / len(returns)


def rank_universe(
    price_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    min_avg_trading_value: float = MIN_AVG_TRADING_VALUE,
) -> pd.DataFrame:
    """여러 종목의 OHLCV(code -> DataFrame, Close/Volume 컬럼 필요)를 받아
    as_of 시점 기준 모멘텀 점수로 순위를 매긴 DataFrame을 반환한다.

    반환 컬럼: 코드, 모멘텀점수, 평균거래대금 (모멘텀점수 내림차순 정렬,
    0번 인덱스가 1위).
    """
    rows = []
    for code, df in price_data.items():
        df_upto = df[df.index <= as_of]
        if df_upto.empty:
            continue

        score = momentum_score(df_upto["Close"], as_of)
        if score is None:
            continue

        recent = df_upto.tail(TRADING_VALUE_WINDOW)
        if len(recent) < TRADING_VALUE_WINDOW:
            continue
        avg_trading_value = float((recent["Close"] * recent["Volume"]).mean())
        if avg_trading_value < min_avg_trading_value:
            continue

        rows.append({"코드": code, "모멘텀점수": score, "평균거래대금": avg_trading_value})

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values("모멘텀점수", ascending=False).reset_index(drop=True)


def select_top_k(ranked: pd.DataFrame, top_k: int) -> list[str]:
    """rank_universe() 결과에서 상위 top_k 종목코드 리스트를 뽑는다."""
    if ranked.empty:
        return []
    return ranked["코드"].head(top_k).tolist()
