"""코스피 지수 일봉으로 "지금 시장이 어떤 상태인지"를 규칙대로 요약한다(시황·전망 화면이 사용).

⚠️ 이건 예측이 아니라 **이미 일어난 가격 움직임을 숫자로 풀어 쓴 것**이다. 문장은 사실 서술에 한정하고
"사라/팔라" 같은 권유나 방향 예측은 쓰지 않는다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from signals.indicators import compute_rsi
from strategy.live_portfolio import current_regime

RSI_PERIOD = 14
VOL_WINDOW = 20


def describe_market(df: pd.DataFrame) -> dict:
    """df: 코스피 지수 일봉(Close 필요, 200일선 계산을 위해 가능하면 1년 이상).

    반환: {"metrics": {...}, "lines": [설명 문장, ...]} — 값이 없으면 그 항목은 None/생략.
    """
    close = df["Close"].dropna()
    if len(close) < 2:
        return {"metrics": {}, "lines": []}
    last = float(close.iloc[-1])
    last_date = close.index[-1]
    year = close[close.index >= last_date - pd.Timedelta(days=365)]
    high, high_date = float(year.max()), year.idxmax()
    drawdown = last / high - 1

    ma200 = close.rolling(200).mean().iloc[-1]
    vs_ma200 = (last / float(ma200) - 1) if pd.notna(ma200) else None
    ret = close.pct_change().dropna()
    r5 = float(close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else None
    r20 = float(close.iloc[-1] / close.iloc[-21] - 1) if len(close) > 20 else None

    vol_now = vol_ratio = None
    if len(ret) >= VOL_WINDOW:
        rolling = ret.rolling(VOL_WINDOW).std().dropna()
        vol_now = float(rolling.iloc[-1])
        typical = float(rolling[rolling.index >= last_date - pd.Timedelta(days=365)].median())
        vol_ratio = vol_now / typical if typical > 0 else None

    rsi = compute_rsi(close, RSI_PERIOD).iloc[-1]
    rsi = float(rsi) if pd.notna(rsi) else None
    mult, regime_label = current_regime({"KOSPI": df})["KOSPI"]

    metrics = {
        "last": last, "date": last_date, "high_1y": high, "high_date": high_date, "drawdown": drawdown,
        "ma200": float(ma200) if pd.notna(ma200) else None, "vs_ma200": vs_ma200, "ret_5d": r5, "ret_20d": r20,
        "vol_20d": vol_now, "vol_ratio": vol_ratio, "rsi": rsi, "regime_mult": mult, "regime_label": regime_label,
    }

    lines = [
        f"코스피 {last:,.0f}는 최근 1년 고점 {high:,.0f}({high_date:%m/%d}) 대비 {drawdown:+.1%}에 있습니다."
    ]
    if vs_ma200 is not None:
        side = "위" if vs_ma200 >= 0 else "아래"
        lines.append(
            f"200일선({float(ma200):,.0f})보다 {abs(vs_ma200):.1%} {side}에 있고, 이 앱의 장세 판단은 "
            f"'{regime_label}'(투자비중 {mult:.0%})입니다."
        )
    if r5 is not None and r20 is not None:
        lines.append(f"최근 5거래일 {r5:+.1%}, 20거래일 {r20:+.1%} 움직였습니다.")
    if vol_now is not None and vol_ratio is not None:
        lines.append(
            f"최근 {VOL_WINDOW}거래일의 하루 등락폭(표준편차)은 {vol_now:.1%}로, 최근 1년 중앙값의 {vol_ratio:.1f}배입니다"
            + (" — 평소보다 크게 출렁이는 구간입니다." if vol_ratio >= 1.5 else ".")
        )
    if rsi is not None:
        lines.append(f"RSI({RSI_PERIOD})는 {rsi:.0f}입니다(통상 30 이하는 과매도, 70 이상은 과매수 구간으로 봅니다).")
    return {"metrics": metrics, "lines": lines}
