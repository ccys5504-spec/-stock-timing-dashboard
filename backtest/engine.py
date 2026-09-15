"""신호 기반 단순 백테스트 엔진 (롱 온리, 전액 매수/매도).

체결 타이밍 원칙 (2026-09-15 재설계):
- 신호는 '그날 종가까지의 데이터'로 계산되므로, 그 신호로 실제 매매할 수 있는
  가장 이른 시점은 **다음 거래일**이다. 신호가 뜬 당일 종가로 즉시 체결하는
  것은 실현 불가능한 가정이므로 쓰지 않는다.
- 매수/매도(신호 기반) 체결가 = 신호가 뜬 날의 **다음 거래일 시가**.
- 손절/트레일링 스탑의 기준선(스탑 가격)은 **전일 종가까지 확정된 값**으로
  정하고, 그 날의 **저가**가 그 선을 건드렸는지로 이탈 여부를 판정한다.
  이탈 시 체결가는 그날 **시가**가 이미 스탑 아래면 시가(갭하락 반영),
  아니면 스탑 가격이다. 당일 고가로 당일 스탑선을 올리는 순환 참조는
  하지 않는다(고가와 저가가 하루 안에서 어느 순서로 발생했는지는 일봉만으로
  알 수 없기 때문).
- 수수료/세금은 왕복 0.3%로 단순화해서 매도 시점에 한 번 반영한다. 거래별
  수익률(`Trade.return_pct`)과 자산곡선(equity curve)은 같은 승수 방식
  `(매도가/매수가) × (1-비용) - 1`을 써서 서로 어긋나지 않게 한다.
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
        # equity curve와 동일한 승수 방식: (매도가/매수가) × (1-비용) - 1.
        # 뺄셈 방식(비율에서 비용을 그냥 빼는 것)은 근사치일 뿐이라 쓰지 않는다.
        return (self.sell_price / self.buy_price) * (1 - ROUND_TRIP_COST) - 1


TRADING_DAYS_PER_YEAR = 252  # 연율화(annualize)에 쓰는 관례적인 국내 주식 거래일수


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    trades: list[Trade] = field(default_factory=list)
    # 그날 포지션을 들고 있었는지(True/False) — exposure(시장 노출도) 계산용.
    # equity_curve와 같은 index 길이. 2026-09-15 Sharpe/Sortino/노출도 추가 때 도입.
    holding_flags: pd.Series | None = None

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
    def daily_returns(self) -> pd.Series:
        """일별 자산곡선 등락률. Sharpe/Sortino 계산의 재료."""
        return self.equity_curve.pct_change().dropna()

    @property
    def sharpe_ratio(self) -> float | None:
        """연율화 샤프 비율 (무위험수익률 0% 가정 — 단순화).

        일별 수익률의 평균/표준편차에 연율화 계수(sqrt(252))를 곱한 값으로,
        '수익의 변동성 대비 크기'를 나타내는 표준 지표다. 절대 수익률만 보면
        변동성이 큰 전략이 유리해 보일 수 있는데, 샤프 비율은 그 위험을
        같이 반영한다. 표본이 너무 적거나(2일 미만) 변동성이 0이면 계산할
        수 없어 None을 반환한다.
        """
        r = self.daily_returns
        if len(r) < 2 or r.std() == 0:
            return None
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def sortino_ratio(self) -> float | None:
        """연율화 소르티노 비율 (무위험수익률 0% 가정).

        샤프 비율과 비슷하지만, '상승 변동성'은 위험으로 치지 않고 '하락
        변동성(손실 방향의 변동성)'만 분모로 쓴다. 트레일링 스탑처럼 상승은
        크게 내버려두고 하락만 방어하려는 전략을 평가할 때 샤프보다 실상을
        더 잘 반영하는 경우가 많다. 하락한 날이 없으면 계산할 수 없어 None.
        """
        r = self.daily_returns
        downside = r[r < 0]
        # downside.std()는 표본이 2개 미만이면 정의상 NaN을 반환한다(표준편차는
        # 최소 2개 값이 있어야 계산 가능) — 하락한 날이 0일이거나 1일뿐이면
        # 소르티노를 믿을 만하게 계산할 수 없으므로 None을 반환한다.
        if len(r) < 2 or len(downside) < 2:
            return None
        downside_std = downside.std()
        if pd.isna(downside_std) or downside_std == 0:
            return None
        return float(r.mean() / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def exposure_pct(self) -> float | None:
        """전체 거래일 중 실제로 주식을 들고 있었던 비율(시장 노출도).

        예를 들어 50%면 절반은 현금(무위험)으로 대기하고 있었다는 뜻이다.
        누적수익률이 같더라도 노출도가 낮은 전략은 '적게 들고도' 같은
        결과를 냈다는 뜻이라 더 효율적이었다고 볼 수 있다.
        """
        if self.holding_flags is None or len(self.holding_flags) == 0:
            return None
        return float(self.holding_flags.mean())

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
    """SIGNAL 컬럼이 포함된 데이터프레임(Open/High/Low/Close 필요)으로
    백테스트를 수행한다.

    stop_loss_pct: 예) 0.1이면 매수가 대비 -10% 하락 시 신호와 무관하게
    손절(당일 시가 또는 스탑가로 청산, 위 모듈 docstring 참고). None이면
    손절 규칙 미적용.

    trailing_stop_pct: 예) 0.15면 보유 중 전일 종가까지의 최고가 대비 -15%
    하락 시 청산한다.

    exit_on_signal: False로 주면 '매도' 신호를 무시하고 손절/트레일링 스탑
    (또는 데이터 끝)으로만 청산한다.

    adaptive_exit: True면 exit_on_signal을 무시하고, 신호가 발생한 날의
    장세(IS_BULL_REGIME)가 상승장이면 '매도' 신호를 무시(트레일링 스탑으로만
    청산)하고 하락장이면 신호를 그대로 따른다. IS_BULL_REGIME 컬럼 필요.
    """
    dates = df_signals.index
    open_ = df_signals["Open"]
    low = df_signals["Low"]
    close = df_signals["Close"]
    signal = df_signals["SIGNAL"]
    is_bull = df_signals["IS_BULL_REGIME"] if adaptive_exit else None

    cash = 1.0  # 초기 자본 = 1 (비율로 취급)
    shares = 0.0
    holding = False
    peak_price = 0.0
    trades: list[Trade] = []
    equity = []
    holding_flags = []

    prev_signal: str | None = None  # 전일까지 확정된 신호 (오늘 시가에 실행)
    prev_is_bull: bool | None = None

    for i in range(len(df_signals)):
        date = dates[i]
        o = open_.iloc[i]
        l = low.iloc[i]
        c = close.iloc[i]

        # 1) 전일 신호를 오늘 시가에 실행 (신호는 하루 지연 체결)
        if not holding and prev_signal == "매수":
            shares = cash / o
            cash = 0.0
            holding = True
            peak_price = o
            trades.append(Trade(buy_date=date, buy_price=o))
        else:
            signal_exit_allowed = exit_on_signal
            if adaptive_exit and prev_is_bull is not None:
                signal_exit_allowed = not prev_is_bull  # 하락장일 때만 신호 매도 허용
            if holding and signal_exit_allowed and prev_signal == "매도":
                cash = shares * o * (1 - ROUND_TRIP_COST)
                shares = 0.0
                holding = False
                trades[-1].sell_date = date
                trades[-1].sell_price = o

        # 2) 보유 중이면 오늘 저가로 손절/트레일링 체크. 기준선은 오늘 시가
        #    진입가 또는 전일 종가까지의 고점이며, 오늘 고가는 쓰지 않는다.
        if holding and stop_loss_pct is not None:
            stop_price = trades[-1].buy_price * (1 - stop_loss_pct)
            if l <= stop_price:
                exit_price = min(o, stop_price)  # 갭하락 시 시가, 아니면 스탑가
                cash = shares * exit_price * (1 - ROUND_TRIP_COST)
                shares = 0.0
                holding = False
                trades[-1].sell_date = date
                trades[-1].sell_price = exit_price
                trades[-1].stopped_out = True

        if holding and trailing_stop_pct is not None:
            trail_price = peak_price * (1 - trailing_stop_pct)
            if l <= trail_price:
                exit_price = min(o, trail_price)
                cash = shares * exit_price * (1 - ROUND_TRIP_COST)
                shares = 0.0
                holding = False
                trades[-1].sell_date = date
                trades[-1].sell_price = exit_price
                trades[-1].trailing_stopped_out = True

        # 3) 보유가 이어지면 오늘 종가로 고점 갱신 (내일 트레일링 기준선에 반영)
        if holding:
            peak_price = max(peak_price, c)

        equity.append(cash + shares * c)
        holding_flags.append(holding)
        prev_signal = signal.iloc[i]
        prev_is_bull = bool(is_bull.iloc[i]) if adaptive_exit else None

    equity_curve = pd.Series(equity, index=dates)
    holding_series = pd.Series(holding_flags, index=dates)
    return BacktestResult(equity_curve=equity_curve, trades=trades, holding_flags=holding_series)


def buy_and_hold_return_pct(df: pd.DataFrame) -> float:
    close = df["Close"]
    return close.iloc[-1] / close.iloc[0] - 1
