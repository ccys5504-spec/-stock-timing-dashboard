"""뷰: 코스피 지수 그래프.

2026-10-16: 예전 "📒 매매 기록·수익률" 탭을 없애고 그 자리에 만들었다. 코스피 지수(KS11)의 일봉을
기간별로 선 그래프(또는 캔들)로 보여주고, 이 앱이 장세 판단(v2 포트폴리오의 투자 비중 배율)에 쓰는
200일선 대비 위치를 함께 보여준다. 시세 출처와 갱신 주기는 다른 탭과 같다(core.load_data, 30분 보관).
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import load_data
from strategy.live_portfolio import current_regime

INDEX_CODE = "KS11"
PERIODS = {"1개월": 31, "3개월": 92, "6개월": 183, "1년": 365, "3년": 1096, "5년": 1826, "10년": 3652}
MA_WINDOWS = {"20일선": 20, "60일선": 60, "200일선": 200}
MA_COLORS = {"20일선": "#f59e0b", "60일선": "#10b981", "200일선": "#ef4444"}


def summarize_index(df: pd.DataFrame, days: int) -> dict:
    """지수 일봉(df, Close 컬럼 필요)에서 화면에 보여줄 숫자를 계산한다.

    이동평균은 기간 시작 이전 데이터까지 포함한 전체 시리즈로 계산해야 기간 맨 앞에서도 값이 있다.
    """
    close = df["Close"].dropna()
    last_date = close.index[-1]
    window = close[close.index >= last_date - pd.Timedelta(days=days)]
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) >= 2 else None
    ma200 = close.rolling(200).mean().iloc[-1]
    return {
        "date": last_date,
        "last": last,
        "day_change": (last / prev - 1) if prev else None,
        "period_return": float(window.iloc[-1] / window.iloc[0] - 1) if len(window) >= 2 else None,
        "high": float(window.max()), "high_date": window.idxmax(),
        "low": float(window.min()), "low_date": window.idxmin(),
        "vs_ma200": (last / float(ma200) - 1) if pd.notna(ma200) else None,
    }


def _build_figure(df: pd.DataFrame, days: int, mas: list[str], candle: bool) -> go.Figure:
    last_date = df.index[-1]
    start = last_date - pd.Timedelta(days=days)
    view = df[df.index >= start]
    fig = go.Figure()
    if candle:
        fig.add_trace(go.Candlestick(
            x=view.index, open=view["Open"], high=view["High"], low=view["Low"], close=view["Close"],
            name="코스피", increasing_line_color="#ef4444", decreasing_line_color="#3b82f6",  # 한국식: 상승 빨강, 하락 파랑
        ))
    else:
        fig.add_trace(go.Scatter(x=view.index, y=view["Close"], name="코스피", mode="lines",
                                 line=dict(width=2.2, color="#6366f1")))
    for name in mas:
        ma = df["Close"].rolling(MA_WINDOWS[name]).mean()
        ma = ma[ma.index >= start]
        fig.add_trace(go.Scatter(x=ma.index, y=ma, name=name, mode="lines",
                                 line=dict(width=1.3, color=MA_COLORS[name], dash="dot")))
    fig.update_layout(
        height=460, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
        yaxis_title="지수", legend=dict(orientation="h", y=-0.12), xaxis_rangeslider_visible=False,
    )
    return fig


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.2%}"


def render() -> None:
    st.caption(
        "코스피 지수(KS11)의 일봉 그래프입니다. 아래 점선은 이동평균선이고, 이 앱은 그중 **200일선**을 "
        "장세 판단(v2 포트폴리오의 투자 비중 조절)에 씁니다. 시세는 다른 탭과 같이 최대 30분 전 값일 수 있습니다."
    )
    c1, c2, c3 = st.columns([2, 2.4, 1.6])
    period = c1.radio("기간", list(PERIODS), index=3, horizontal=True, key="idx_period")
    mas = c2.multiselect("이동평균선", list(MA_WINDOWS), default=["200일선"], key="idx_mas")
    chart_type = c3.radio("그래프", ["선", "캔들"], horizontal=True, key="idx_chart")
    days = PERIODS[period]

    years = max(1, int(days / 365.25) + 1)  # 기간보다 조금 넉넉하게(이동평균 계산용 앞부분은 load_data가 따로 더 받아온다)
    try:
        df = load_data(INDEX_CODE, years)
    except Exception as e:  # noqa: BLE001 — 시세 조회 실패를 화면에 그대로 알린다
        st.error(f"코스피 지수를 불러오지 못했습니다: {type(e).__name__}: {e}")
        return
    if df is None or df.empty:
        st.warning("코스피 지수 데이터가 비어 있습니다. 잠시 뒤 다시 열어보세요.")
        return

    s = summarize_index(df, days)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(f"코스피 ({s['date'].date().isoformat()} 기준)", f"{s['last']:,.2f}",
              _pct(s["day_change"]) if s["day_change"] is not None else None)
    m2.metric(f"최근 {period} 수익률", _pct(s["period_return"]))
    m3.metric("200일선 대비", _pct(s["vs_ma200"]),
              help="현재 지수가 200일 이동평균선보다 얼마나 위(+)/아래(−)에 있는지")
    mult, label = current_regime({"KOSPI": df})["KOSPI"]
    m4.metric("이 앱의 장세 판단", f"투자비중 {mult:.0%}", help=label)

    st.plotly_chart(_build_figure(df, days, mas, chart_type == "캔들"), use_container_width=True)
    st.caption(
        f"기간 최고 {s['high']:,.2f} ({s['high_date'].date().isoformat()}) · "
        f"최저 {s['low']:,.2f} ({s['low_date'].date().isoformat()}) · 장세 판단: {label}"
    )
    st.caption("지수 시세 출처는 종목 시세와 같은 FinanceDataReader이며, 투자 자문이 아닌 참고용 그래프입니다.")
