"""전략 성과를 비교할 기준선(벤치마크) 계산.

2026-09-20 점검서 지적: KOSPI 지수 단순보유 하나만 비교하면 "이 전략이 KOSPI를
이겼다"가 곧 "전략이 좋다"로 읽힌다. 그런데 v2가 고르는 종목군(오늘 시점 시가총액
상위 100개) 자체가 KOSPI보다 훨씬 많이 올랐기 때문에, 진짜 비교 대상은 **같은
종목군을 그냥 동일가중으로 들고 있었을 때**다. 이 모듈이 그 기준선을 계산한다.

⚠️ 이 동일가중 보유도 오늘 시점의 승자만 모은 종목군이라 생존편향이 그대로 섞여
있다 — 실제로 투자할 수 있었던 대안이 아니라, 생존편향이 얼마나 큰지 가늠하는
상한선으로 읽어야 한다.
"""
from __future__ import annotations

import pandas as pd

from strategy.momentum import rank_universe


def equal_weight_equity(
    price_data: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp
) -> pd.Series:
    """매일 동일가중으로 재조정하는 포트폴리오의 자산곡선(시작=1.0).

    그날 종가가 있는 종목들의 일간 수익률을 단순평균한다 — 상장 전 종목은
    자동으로 빠지고 상장 이후부터 편입된다. 비용은 반영하지 않는다(보유
    기준선이므로 전략 쪽 비용과 공정하게 비교하려면 이 수치가 "비용 0"이라는
    점을 감안해야 한다).
    """
    close = pd.DataFrame({code: df["Close"] for code, df in price_data.items()})
    close = close[(close.index >= start) & (close.index <= end)]
    if close.empty:
        raise ValueError("해당 기간에 가격 데이터가 없습니다.")
    # 거래정지 등으로 빈 날은 직전 종가로 이어 붙인다(예전 pct_change 기본 동작과 동일).
    # 상장 전 구간의 NaN은 그대로 남아 평균에서 자동으로 빠진다.
    daily = close.ffill().pct_change(fill_method=None).mean(axis=1, skipna=True).fillna(0.0)
    return (1 + daily).cumprod()


def summarize_equity(equity: pd.Series) -> tuple[float, float]:
    """(누적수익률, 최대낙폭)."""
    total = float(equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 1 else 0.0
    mdd = float((equity / equity.cummax() - 1).min())
    return total, mdd


def pit_equal_weight_equity(
    price_data: dict[str, pd.DataFrame], calendar: pd.DatetimeIndex, top_n: int,
    shares: dict[str, float] | None = None,
) -> pd.Series:
    """시점 기준(PIT) 동일가중 기준선: 매월 말 그날의 거래대금 상위 top_n개를 뽑아(상장폐지될
    종목 포함) 다음 달 동안 동일가중으로 '그대로 보유'한다(월 안에서는 재조정 안 함).
    시세가 끝난 종목(폐지)은 마지막 종가에서 현금화한 것으로 본다. 비용은 반영하지 않는다.

    v2 전략이 종목 선택(모멘텀·손절)으로 실제로 값어치를 더했는지를 "같은 후보군을 그냥
    동일가중으로 들고 있었을 때"와 공정하게(생존편향 없이) 비교하기 위한 기준선이다.
    """
    close = pd.DataFrame({c: d["Close"] for c, d in price_data.items()}).reindex(calendar).ffill()
    month_key = calendar.to_period("M")
    month_ends = sorted(set(pd.Series(calendar, index=calendar).groupby(month_key).max().tolist()))
    equity = pd.Series(1.0, index=calendar)
    level = 1.0
    start_pos = 0
    held: list[str] = []
    boundaries = [calendar.get_loc(d) for d in month_ends]
    # 각 월말(리밸런싱) 다음 날부터 다음 월말까지: 그 월말 종가에 동일가중으로 사서 그대로 보유
    for k, end_pos in enumerate(boundaries):
        if held:
            seg_start = start_pos  # 직전 리밸런싱일
            base = close.iloc[seg_start][held]
            path = close.iloc[seg_start + 1: end_pos + 1][held].div(base).mean(axis=1, skipna=True)
            equity.iloc[seg_start + 1: end_pos + 1] = level * path.values
            if len(path):
                level = float(level * path.iloc[-1])
        elif end_pos > 0:
            equity.iloc[: end_pos + 1] = level
        start_pos = end_pos
        ranked = rank_universe(price_data, calendar[end_pos], top_n=top_n, shares=shares)
        held = ranked["코드"].tolist() if not ranked.empty else []  # 후보가 없으면 현금
    if start_pos + 1 < len(calendar):
        equity.iloc[start_pos + 1:] = level
    return equity
