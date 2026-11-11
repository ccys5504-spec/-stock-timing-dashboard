"""사용자가 정한 매도 규칙(손절 -3%, 이익 구간별로 올라가는 매도선)을 계산한다.

규칙(2026-11-11 사용자 지정, 해석은 아래 그대로):
  - 처음에는 매입가 대비 -3%가 매도선이다(그 가격에 닿으면 매도).
  - 고점 수익률이 +5% / +10% / +20%를 넘어서면, 그 구간의 가격(+5% / +10% / +20%)이 새 매도선이 된다.
    예) +12%까지 올랐다가 +10%로 내려오면 매도, +25%까지 올랐다가 +20%로 내려오면 매도.
  - 매도선은 한 번 올라가면 내려가지 않는다. +20%를 넘은 뒤에는 매도선이 +20%에 머문다
    (더 올라도 +30% 같은 새 구간은 없음 — 필요하면 MILESTONES에 추가).

일봉으로 판정하는 방법(같은 날 안의 고가/저가 순서는 알 수 없으므로 보수적으로):
  - 그날의 매도선은 '전날까지의 고점'으로 정한다(당일 고점은 다음 날부터 반영).
  - 시가가 매도선 아래로 열리면(갭하락) 시가에, 장중에 저가가 매도선에 닿으면 매도선 가격에 체결된 것으로 본다.
    실제로는 갭하락이나 하한가에서 매도선보다 훨씬 낮게 체결되거나 못 팔 수도 있다.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class ExitRule:
    stop_loss: float = 0.03                      # 매입가 대비 하락률(양수로 적는다)
    milestones: tuple[float, ...] = (0.05, 0.10, 0.20)

    def floor_return(self, peak_return: float) -> float:
        """고점 수익률(peak_return)에서 지금 유효한 매도선의 수익률(예: -0.03, 0.05, 0.10, 0.20)."""
        reached = [m for m in self.milestones if peak_return >= m]
        return max(reached) if reached else -self.stop_loss

    def label(self, floor_return: float) -> str:
        return f"손절 {floor_return:+.0%}" if floor_return < 0 else f"이익 보호 {floor_return:+.0%}"


DEFAULT_RULE = ExitRule()


@dataclass(frozen=True)
class ExitResult:
    triggered: bool
    exit_index: int | None       # 매도 조건이 처음 충족된 봉의 위치(0 = 첫 봉)
    exit_date: pd.Timestamp | None
    exit_price: float | None
    exit_floor: float | None     # 발동 당시의 매도선 수익률
    peak_return: float           # 마지막 판정 시점까지의 고점 수익률(발동 시에는 발동 직전까지)
    floor_return: float          # 마지막 판정 시점의 매도선 수익률
    last_return: float           # 마지막 종가 수익률


def simulate_exit(bars: pd.DataFrame, entry_price: float, rule: ExitRule = DEFAULT_RULE) -> ExitResult:
    """bars: 매수한 날부터의 일봉(Open/High/Low/Close, 첫 봉 = 매수일, 시가에 entry_price로 샀다고 가정).

    첫 봉부터 순서대로 보면서 매도 조건이 처음 충족되는 봉을 찾는다. 충족되지 않으면 triggered=False.
    """
    if entry_price <= 0:
        raise ValueError("entry_price는 0보다 커야 합니다.")
    peak = 0.0
    for i, (date, row) in enumerate(bars.iterrows()):
        floor = rule.floor_return(peak)
        floor_price = entry_price * (1 + floor)
        o, h, l = float(row["Open"]), float(row["High"]), float(row["Low"])
        if min(o, h, l) <= 0 or pd.isna(o) or pd.isna(h) or pd.isna(l):
            continue  # 거래정지일 등으로 가격이 0/빈값으로 들어온 봉은 거래가 없던 날로 보고 건너뛴다(가짜 폭락 방지)
        if i > 0 and o <= floor_price:          # 갭하락으로 시작: 시가에 판다(첫 봉은 시가=매입가라 해당 없음)
            return ExitResult(True, i, date, o, floor, peak, floor, o / entry_price - 1)
        if l <= floor_price:                     # 장중에 매도선에 닿음: 매도선 가격에 판다
            return ExitResult(True, i, date, floor_price, floor, peak, floor, floor_price / entry_price - 1)
        peak = max(peak, h / entry_price - 1)
    closes = bars["Close"][bars["Close"] > 0]
    last = float(closes.iloc[-1]) / entry_price - 1 if len(closes) else 0.0
    return ExitResult(False, None, None, None, None, peak, rule.floor_return(peak), last)


def evaluate_holding(
    df: pd.DataFrame, entry_price: float, entry_date: str | pd.Timestamp | None, rule: ExitRule = DEFAULT_RULE,
) -> dict:
    """보유 종목 한 개의 현재 상태. df는 해당 종목의 일봉(Open/High/Low/Close)이다.

    entry_date를 모르면(매수일 미입력) 가장 최근 봉 하나만으로 판정한다 — 그 전의 고점을 알 수 없어서
    이익 구간 매도선이 실제보다 낮게 나올 수 있다(precise=False).
    """
    if df is None or df.empty:
        raise ValueError("시세가 없습니다.")
    precise = entry_date is not None and str(entry_date).strip() != ""
    if precise:
        bars = df[df.index >= pd.Timestamp(entry_date)]
        if bars.empty:
            raise ValueError("매수일 이후의 시세가 없습니다.")
    else:
        bars = df.iloc[[-1]]
    res = simulate_exit(bars, entry_price, rule)
    valid_close = bars["Close"][bars["Close"] > 0]
    last_close = float(valid_close.iloc[-1]) if len(valid_close) else float(bars["Close"].iloc[-1])
    return {
        "triggered": res.triggered, "exit_date": res.exit_date, "exit_price": res.exit_price,
        "peak_return": res.peak_return, "floor_return": res.floor_return,
        "floor_price": entry_price * (1 + res.floor_return),
        "last_price": last_close, "last_return": last_close / entry_price - 1,
        "last_date": bars.index[-1], "precise": precise,
        "label": rule.label(res.exit_floor if res.triggered and res.exit_floor is not None else res.floor_return),
    }
