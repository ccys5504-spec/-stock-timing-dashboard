"""기술적 지표 계산 및 매수/매도 점수 신호 로직.

4개 지표(이평선 골든/데드크로스, RSI, MACD, 볼린저밴드)를 각각 -1/0/+1로 채점해
합산한다.

참고: 4개 지표를 모두 요구(임계값 ±2 이상)하면 실제 백테스트에서 3년에 1회
수준으로 거의 신호가 나지 않았다(각 지표의 '크로스' 이벤트가 같은 날 동시에
겹치는 경우가 매우 드물기 때문). 반대로 지표 1개 신호만으로도 인정(±1)하면
종목당 월 1회 안팎으로, 단타는 아니면서 주기적으로 매매하는 빈도가 나왔다.
이에 따라 기본값은 ±1로 설정했다. 더 신중하게(신호를 줄이고 싶다면) ±2로,
더 공격적으로(신호를 늘리고 싶다면) 개별 지표 조건을 완화하면 된다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ---- 파라미터 (필요시 조절) ----
MA_SHORT = 20
MA_LONG = 60
RSI_PERIOD = 14
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2

BUY_THRESHOLD = 1
SELL_THRESHOLD = -1

# ---- 거래량 필터 (선택) ----
# 신호가 뜬 날 거래량이 최근 평균보다 충분히 커야("관심이 실린 움직임") 신호를
# 인정한다. 얇은 거래량에서 발생하는 잡음성 신호를 걸러내기 위함.
# volume_filter 값은 False(끔) / True(항상 켬) / "auto"(아래 REGIME 기준 자동) 중 하나.
VOLUME_FILTER = False
VOLUME_PERIOD = 20
VOLUME_MULTIPLIER = 1.2

# ---- 상승장/하락장 자동 판별 (거래량 필터 "auto" 모드에서 사용) ----
# 종가가 장기 이동평균(기본 200일) 위에 있으면 "상승장", 아래면 "하락장"으로
# 본다. 실제 A/B 테스트 결과(README 참고), 거래량 필터는 하락장 방어에는
# 거의 항상 도움이 되지만 상승장에서는 수익을 크게 깎는 경우가 많았다.
# 그래서 auto 모드는 "하락장일 때만 거래량 필터를 켠다".
REGIME_MA_PERIOD = 200


def _rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # avg_loss=0인데 avg_gain>0이면(구간 내내 상승만) RSI는 정의상 100이어야
    # 한다. 위 나눗셈은 이 경우 avg_loss를 NaN으로 바꿔 RSI도 NaN이 되므로
    # 명시적으로 100을 채운다. avg_gain도 0(가격 변화 자체가 없음)인 경우만
    # 진짜 미정의 상태로 보고 중립값 50을 채운다.
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return rsi.fillna(50)


def add_indicators(
    df: pd.DataFrame,
    ma_short: int = MA_SHORT,
    ma_long: int = MA_LONG,
    rsi_period: int = RSI_PERIOD,
    macd_fast: int = MACD_FAST,
    macd_slow: int = MACD_SLOW,
    macd_signal: int = MACD_SIGNAL,
    bb_period: int = BB_PERIOD,
    bb_std: float = BB_STD,
    regime_ma_period: int = REGIME_MA_PERIOD,
) -> pd.DataFrame:
    """OHLCV 데이터프레임에 지표 컬럼들을 추가해서 반환한다.

    ma_short/ma_long 등을 넘기면 그 기간으로 계산한다 (기본값은 모듈 상단 상수).
    파라미터 탐색(scripts/optimize.py)에서 여러 기간 조합을 테스트할 때 쓰인다.
    """
    out = df.copy()
    close = out["Close"]

    # 상승장/하락장 자동 판별용 (거래량 필터 "auto" 모드에서 사용)
    regime_ma = close.rolling(regime_ma_period).mean()
    out["REGIME_MA"] = regime_ma
    out["IS_BULL_REGIME"] = close > regime_ma

    out["MA_SHORT"] = close.rolling(ma_short).mean()
    out["MA_LONG"] = close.rolling(ma_long).mean()

    out["RSI"] = _rsi(close, rsi_period)

    ema_fast = close.ewm(span=macd_fast, adjust=False).mean()
    ema_slow = close.ewm(span=macd_slow, adjust=False).mean()
    out["MACD"] = ema_fast - ema_slow
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=macd_signal, adjust=False).mean()

    bb_mid = close.rolling(bb_period).mean()
    bb_std_series = close.rolling(bb_period).std()
    out["BB_MID"] = bb_mid
    out["BB_UPPER"] = bb_mid + bb_std * bb_std_series
    out["BB_LOWER"] = bb_mid - bb_std * bb_std_series

    return out


def add_scores(
    df: pd.DataFrame,
    buy_threshold: int | None = None,
    sell_threshold: int | None = None,
    volume_filter: bool | str | None = None,
    volume_period: int = VOLUME_PERIOD,
    volume_multiplier: float = VOLUME_MULTIPLIER,
    rsi_oversold: int | None = None,
    rsi_overbought: int | None = None,
) -> pd.DataFrame:
    """지표 컬럼이 채워진 데이터프레임에 개별 점수와 합산 점수/신호를 추가한다.

    buy_threshold/sell_threshold/volume_filter/rsi_oversold/rsi_overbought를
    넘기지 않으면 모듈 상단의 기본값(BUY_THRESHOLD 등)을 사용한다. 파라미터
    탐색(scripts/optimize.py) 등에서 전역 상태를 건드리지 않고 다른 조합을
    테스트할 때 인자로 넘기면 된다.

    volume_filter: False(끔) / True(항상 켬) / "auto"(하락장일 때만 자동으로 켬,
    IS_BULL_REGIME 컬럼 필요 — add_indicators가 만들어둔다).
    """
    buy_threshold = BUY_THRESHOLD if buy_threshold is None else buy_threshold
    sell_threshold = SELL_THRESHOLD if sell_threshold is None else sell_threshold
    volume_filter = VOLUME_FILTER if volume_filter is None else volume_filter
    rsi_oversold = RSI_OVERSOLD if rsi_oversold is None else rsi_oversold
    rsi_overbought = RSI_OVERBOUGHT if rsi_overbought is None else rsi_overbought

    out = df.copy()
    close = out["Close"]
    prev_close = close.shift(1)

    # 1) 이평선 골든/데드크로스
    ma_diff = out["MA_SHORT"] - out["MA_LONG"]
    prev_ma_diff = ma_diff.shift(1)
    ma_score = pd.Series(0, index=out.index, dtype=int)
    ma_score[(prev_ma_diff <= 0) & (ma_diff > 0)] = 1
    ma_score[(prev_ma_diff >= 0) & (ma_diff < 0)] = -1
    out["SCORE_MA"] = ma_score

    # 2) RSI 과매도/과매수
    rsi_score = pd.Series(0, index=out.index, dtype=int)
    rsi_score[out["RSI"] <= rsi_oversold] = 1
    rsi_score[out["RSI"] >= rsi_overbought] = -1
    out["SCORE_RSI"] = rsi_score

    # 3) MACD 크로스
    macd_diff = out["MACD"] - out["MACD_SIGNAL"]
    prev_macd_diff = macd_diff.shift(1)
    macd_score = pd.Series(0, index=out.index, dtype=int)
    macd_score[(prev_macd_diff <= 0) & (macd_diff > 0)] = 1
    macd_score[(prev_macd_diff >= 0) & (macd_diff < 0)] = -1
    out["SCORE_MACD"] = macd_score

    # 4) 볼린저밴드 하단 터치 후 반등 / 상단 터치 후 하락
    prev_low = out["Low"].shift(1)
    prev_high = out["High"].shift(1)
    prev_bb_lower = out["BB_LOWER"].shift(1)
    prev_bb_upper = out["BB_UPPER"].shift(1)
    bb_score = pd.Series(0, index=out.index, dtype=int)
    bb_score[(prev_low <= prev_bb_lower) & (close > prev_close)] = 1
    bb_score[(prev_high >= prev_bb_upper) & (close < prev_close)] = -1
    out["SCORE_BB"] = bb_score

    out["SCORE_TOTAL"] = (
        out["SCORE_MA"] + out["SCORE_RSI"] + out["SCORE_MACD"] + out["SCORE_BB"]
    )

    # 5) 거래량 확인: 최근 N일 평균 거래량 대비 일정 배수 이상일 때만 "확인됨"
    vol_ma = out["Volume"].rolling(volume_period).mean()
    out["VOLUME_OK"] = out["Volume"] > vol_ma * volume_multiplier

    raw_buy = out["SCORE_TOTAL"] >= buy_threshold
    raw_sell = out["SCORE_TOTAL"] <= sell_threshold
    if volume_filter == "auto":
        # 하락장(장기추세 아래)일 때만 거래량 확인을 요구, 상승장에서는 요구하지 않음
        require_volume = ~out["IS_BULL_REGIME"]
        raw_buy &= out["VOLUME_OK"] | ~require_volume
        raw_sell &= out["VOLUME_OK"] | ~require_volume
    elif volume_filter:
        raw_buy &= out["VOLUME_OK"]
        raw_sell &= out["VOLUME_OK"]

    signal = pd.Series("관망", index=out.index)
    signal[raw_buy] = "매수"
    signal[raw_sell] = "매도"
    out["SIGNAL"] = signal

    return out


def build_signals(
    df: pd.DataFrame,
    ma_short: int = MA_SHORT,
    ma_long: int = MA_LONG,
    rsi_period: int = RSI_PERIOD,
    rsi_oversold: int | None = None,
    rsi_overbought: int | None = None,
    buy_threshold: int | None = None,
    sell_threshold: int | None = None,
    volume_filter: bool | str | None = None,
    volume_period: int = VOLUME_PERIOD,
    volume_multiplier: float = VOLUME_MULTIPLIER,
    regime_ma_period: int = REGIME_MA_PERIOD,
) -> pd.DataFrame:
    """OHLCV -> 지표 + 점수 + 신호까지 한 번에 계산.

    ma_short/ma_long/rsi_period은 지표 계산 자체에, 나머지는 점수/신호 판정에
    쓰인다. 전부 생략하면 모듈 기본값을 쓴다. volume_filter="auto"면
    regime_ma_period(기본 200일)로 상승장/하락장을 자동 판별해서 하락장일 때만
    거래량 필터를 적용한다.
    """
    ind_df = add_indicators(
        df, ma_short=ma_short, ma_long=ma_long, rsi_period=rsi_period,
        regime_ma_period=regime_ma_period,
    )
    return add_scores(
        ind_df,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        volume_filter=volume_filter,
        volume_period=volume_period,
        volume_multiplier=volume_multiplier,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
    )
