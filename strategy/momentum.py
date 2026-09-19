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


# as_of 시점에 마지막 시세가 이보다 오래된 종목은 "이미 상장폐지됐거나 오래 거래정지"로 보고
# 순위에서 뺀다. 이 검사가 없으면 상장폐지된 종목의 마지막 시세가 영원히 "최근 20일"로
# 취급돼 죽은 종목이 계속 후보로 뽑힌다(2026-09-20, 생존편향 제거 작업 중 추가).
MAX_STALE_DAYS = 10

# 시점 기준 유니버스(top_n)를 고르는 잣대: 최근 120거래일 거래대금의 '중앙값'.
# 처음엔 최근 20일 '평균'으로 뽑았더니 며칠 급등하며 거래가 폭발한 테마주(파인디앤씨·
# 고려산업 등)가 상위 100개를 채워서, 대형주가 아니라 투기성 종목군이 됐고 전략이
# -90%까지 무너졌다(2026-09-20 생존편향 검증 중 확인). 중앙값은 일시적 폭증에 안 끌려가고
# 꾸준히 거래되는 종목(=대체로 대형주)을 고른다.
UNIVERSE_VALUE_WINDOW = 120
UNIVERSE_MIN_DAYS = 60


def rank_universe(
    price_data: dict[str, pd.DataFrame],
    as_of: pd.Timestamp,
    min_avg_trading_value: float = MIN_AVG_TRADING_VALUE,
    top_n: int | None = None,
    max_stale_days: int = MAX_STALE_DAYS,
    shares: dict[str, float] | None = None,
) -> pd.DataFrame:
    """여러 종목의 OHLCV(code -> DataFrame, Close/Volume 컬럼 필요)를 받아
    as_of 시점 기준 모멘텀 점수로 순위를 매긴 DataFrame을 반환한다.

    top_n을 주면 후보를 상위 top_n개로 좁힌 뒤 모멘텀 순위를 매긴다. shares(종목코드->상장주식수)를
    주면 그 시점 시가총액 근사(보정종가 x 주식수) 상위, 안 주면 최근 120거래일 거래대금 중앙값 상위다
    (거래량이 분할/병합에 보정되지 않는 데이터에서는 shares 방식을 쓸 것) — "오늘 시가총액 상위"가 아니라 "그날 실제로 활발히
    거래되던 상위 종목"이라는 시점 기준(point-in-time) 유니버스다. 시세가 그날
    존재했던 종목만 대상이므로, 나중에 상장폐지된 종목도 폐지 전에는 후보가 된다.

    반환 컬럼: 코드, 모멘텀점수, 평균거래대금 (모멘텀점수 내림차순 정렬,
    0번 인덱스가 1위).
    """
    stale_cutoff = as_of - pd.Timedelta(days=max_stale_days)
    liquid: list[tuple[str, float, pd.DataFrame]] = []
    for code, df in price_data.items():
        idx = df.index
        if len(idx) < TRADING_VALUE_WINDOW or idx[0] > as_of:
            continue  # 데이터가 모자라거나 아직 상장 전
        n = idx.searchsorted(as_of, side="right")
        if n < TRADING_VALUE_WINDOW:
            continue
        df_upto = df.iloc[:n]
        if idx[n - 1] < stale_cutoff:
            continue  # 시세가 이미 끊긴(폐지/장기 정지) 종목
        recent = df_upto.iloc[-TRADING_VALUE_WINDOW:]
        avg_trading_value = float((recent["Close"] * recent["Volume"]).mean())
        if avg_trading_value < min_avg_trading_value:
            continue
        liquid.append((code, avg_trading_value, df_upto))

    if top_n is not None and shares is not None:
        # 시가총액 근사 = 그 시점 보정종가 x 상장주식수. 거래량이 보정되지 않는 데이터에서는
        # 거래대금보다 훨씬 믿을 만하다(build_pit_dataset.py 설명 참고).
        capped = [
            (float(df_upto["Close"].iloc[-1]) * shares[code], code, avg_value, df_upto)
            for code, avg_value, df_upto in liquid if shares.get(code) and shares[code] > 0
        ]
        capped.sort(key=lambda x: x[0], reverse=True)
        liquid = [(code, avg_value, df_upto) for _, code, avg_value, df_upto in capped[:top_n]]
    elif top_n is not None:
        scored = []
        for code, avg_value, df_upto in liquid:
            window = df_upto.iloc[-UNIVERSE_VALUE_WINDOW:]
            if len(window) < UNIVERSE_MIN_DAYS:
                continue
            scored.append((float((window["Close"] * window["Volume"]).median()), code, avg_value, df_upto))
        scored.sort(key=lambda x: x[0], reverse=True)
        liquid = [(code, avg_value, df_upto) for _, code, avg_value, df_upto in scored[:top_n]]

    rows = []
    for code, avg_trading_value, df_upto in liquid:
        score = momentum_score(df_upto["Close"], as_of)
        if score is None:
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
