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
