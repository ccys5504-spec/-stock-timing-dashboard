"""신호 기반 단순 백테스트 엔진 (롱 온리, 전액 매수/매도).

- '매수' 신호 + 미보유 상태 -> 다음날 종가로 매수
- '매도' 신호 + 보유 상태 -> 다음날 종가로 매도
- 수수료/세금은 단순화를 위해 왕복 0.3%로 고정 반영
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ROUND_TRIP_COST = 0.003  # 매수+매도 수수료·세금 합산 근사치


@dataclass
class Trade:
    buy_date: pd.Timestamp
    buy_price: float
    sell_date: pd.Timestamp | None = None
    sell_price: float | None = None
    stopped_out: bool = False  # 신호가 아니라 손절 규칙으로 청산되었는지
    trailing_stopped_out: bool = False  # 트레일링 스탑으로 청산되었는지

    @property
    def return_pct(self) -> float | None:
        if self.sell_price is None:
            return None
        return (self.sell_price / self.buy_price - 1) - ROUND_TRIP_COST


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1

    @property
    def mdd_pct(self) -> float:
        cummax = self.equity_curve.cummax()
        drawdown = self.equity_curve / cummax - 1
        return drawdown.min()

    @property
    def num_trades(self) -> int:
        return len([t for t in self.trades if t.sell_price is not None])

    @property
    def win_rate_pct(self) -> float | None:
        closed = [t.return_pct for t in self.trades if t.return_pct is not None]
        if not closed:
            return None
        wins = sum(1 for r in closed if r > 0)
        return wins / len(closed)

    @property
    def avg_holding_days(self) -> float | None:
        closed = [t for t in self.trades if t.sell_date is not None]
        if not closed:
            return None
        days = [(t.sell_date - t.buy_date).days for t in closed]
        return float(np.mean(days))

    @property
    def num_stopped_out(self) -> int:
        return sum(1 for t in self.trades if t.stopped_out)

    @property
    def num_trailing_stopped_out(self) -> int:
        return sum(1 for t in self.trades if t.trailing_stopped_out)


def run_backtest(
    df_signals: pd.DataFrame,
    stop_loss_pct: float | None = None,
    trailing_stop_pct: float | None = None,
    exit_on_signal: bool = True,
    adaptive_exit: bool = False,
) -> BacktestResult:
    """SIGNAL 컬럼이 포함된 데이터프레임으로 백테스트를 수행한다.

    stop_loss_pct: 예) 0.1 이면 매수가 대비 -10% 하락 시 신호와 무관하게
    즉시 손절(그날 종가로 청산). None이면 손절 규칙 미적용.

    trailing_stop_pct: 예) 0.15면 보유 중 기록한 최고가 대비 -15% 하락 시
    청산한다 (매입가가 아니라 그동안의 최고가 기준이라 이익을 어느 정도
    지키면서도 추세가 계속되는 동안은 계속 들고 갈 수 있다).

    exit_on_signal: False로 주면 '매도' 신호를 무시하고 손절/트레일링 스탑
    (또는 데이터 끝)으로만 청산한다 — 지표가 너무 일찍 파는지 확인할 때 쓴다.

    adaptive_exit: True면 exit_on_signal을 무시하고, 상승장(IS_BULL_REGIME)일
    때는 '매도' 신호를 무시(트레일링 스탑으로만 청산 — 추세를 오래 탐)하고,
    하락장일 때는 '매도' 신호를 그대로 따른다(방어 우선). IS_BULL_REGIME
    컬럼 필요(add_indicators가 만들어둠).
    """
    dates = df_signals.index
    close = df_signals["Close"]
    low = df_signals["Low"]
    signal = df_signals["SIGNAL"]
    is_bull = df_signals["IS_BULL_REGIME"] if adaptive_exit else None

    cash = 1.0  # 초기 자본 = 1 (비율로 취급)
    shares = 0.0
    holding = False
    peak_price = 0.0
    trades: list[Trade] = []
    equity = []

    for i in range(len(df_signals)):
        date = dates[i]
        price = close.iloc[i]
        sig = signal.iloc[i]

        if holding:
            peak_price = max(peak_price, price)

        # 손절/트레일링 스탑 체크가 매수/매도 신호보다 우선
        if holding and stop_loss_pct is not None:
            stop_price = trades[-1].buy_price * (1 - stop_loss_pct)
            if low.iloc[i] <= stop_price:
                exit_price = min(price, stop_price)  # 갭하락 시 보수적으로 반영
                cash = shares * exit_price * (1 - ROUND_TRIP_COST)
                shares = 0.0
                holding = False
                trades[-1].sell_date = date
                trades[-1].sell_price = exit_price
                trades[-1].stopped_out = True
                equity.append(cash)
                continue

        if holding and trailing_stop_pct is not None:
            trail_price = peak_price * (1 - trailing_stop_pct)
            if low.iloc[i] <= trail_price:
                exit_price = min(price, trail_price)
                cash = shares * exit_price * (1 - ROUND_TRIP_COST)
                shares = 0.0
                holding = False
                trades[-1].sell_date = date
                trades[-1].sell_price = exit_price
                trades[-1].trailing_stopped_out = True
                equity.append(cash)
                continue

        signal_exit_allowed = exit_on_signal
        if adaptive_exit:
            signal_exit_allowed = not bool(is_bull.iloc[i])  # 하락장일 때만 신호 매도 허용

        if not holding and sig == "매수":
            shares = cash / price
            cash = 0.0
            holding = True
            peak_price = price
            trades.append(Trade(buy_date=date, buy_price=price))
        elif holding and signal_exit_allowed and sig == "매도":
            cash = shares * price * (1 - ROUND_TRIP_COST)
            shares = 0.0
            holding = False
            trades[-1].sell_date = date
            trades[-1].sell_price = price  # 비용은 return_pct에서 별도 반영

        equity.append(cash + shares * price)

    equity_curve = pd.Series(equity, index=dates)
    return BacktestResult(equity_curve=equity_curve, trades=trades)


def buy_and_hold_return_pct(df: pd.DataFrame) -> float:
    close = df["Close"]
    return close.iloc[-1] / close.iloc[0] - 1
