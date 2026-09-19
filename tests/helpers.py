"""테스트용 합성 데이터 생성기 (네트워크/실제 시세 사용 안 함)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ohlcv(dates: pd.DatetimeIndex, seed: int = 0, drift: float = 0.0004, vol: float = 0.015,
               start_price: float = 10_000.0, volume: float = 2_000_000.0) -> pd.DataFrame:
    """랜덤워크 기반 OHLCV. 시가는 전일 종가 근처, 고가/저가는 시가·종가를 감싼다."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, vol, len(dates))
    close = start_price * np.cumprod(1 + rets)
    open_ = np.concatenate([[start_price], close[:-1]]) * (1 + rng.normal(0, 0.003, len(dates)))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, len(dates))))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, len(dates))))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close,
         "Volume": np.full(len(dates), volume)},
        index=dates,
    )


def make_signal_frame(opens, lows, closes, signals) -> pd.DataFrame:
    """v1 엔진(run_backtest) 입력용 최소 데이터프레임."""
    idx = pd.bdate_range("2024-01-01", periods=len(opens))
    return pd.DataFrame({"Open": opens, "Low": lows, "Close": closes, "SIGNAL": signals}, index=idx)
