"""뷰 1: 단일 종목 분석."""
from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.engine import buy_and_hold_return_pct, run_backtest
from core import CHART_TIMEFRAMES, Settings, chart_dataframe, prepare
from data.fetch import load_holdings, resolve_stock_name
from signals import indicators as ind
from views.order_link import render_order_link


def _holdings_options() -> dict[str, str]:
    """보유종목 목록에서 {종목명: 종목코드}를 만든다(중복 종목코드는 한 번만)."""
    result: dict[str, str] = {}
    for h in load_holdings():
        code = str(h.get("종목코드", "")).strip().zfill(6)
        if not code or code in result.values():
            continue
        result[resolve_stock_name(code) or code] = code
    return result


def render(watchlist: dict[str, str], years: int, settings: Settings) -> None:
    jump = st.session_state["jump"]
    # 2026-09-18: 기본 선택지를 관심종목 대신 "보유종목"으로 바꿨다 — 실제로
    # 갖고 있는 종목을 살펴보는 게 더 자주 쓰는 용도라서다. 다만 보유종목을
    # 아직 하나도 안 넣었으면 고를 게 없어지므로, 그럴 때만 관심종목으로
    # 대신한다. 보유하지 않은 새 종목을 보고 싶으면 '종목 추천(스크리너)'
    # 탭에서 결과를 클릭해서 넘어오면 된다(jump).
    held = _holdings_options()
    base = held if held else watchlist
    options = list(base.keys())
    if jump and jump[0] not in options:
        options = [jump[0]] + options
    default_index = options.index(jump[0]) if jump and jump[0] in options else 0

    label = "종목 선택 (보유종목 + 스크리너에서 넘어온 종목)" if held else \
        "종목 선택 (관심종목 + 스크리너에서 넘어온 종목 — 보유종목을 입력하면 그게 먼저 보입니다)"
    name = st.selectbox(label, options, index=default_index)
    code = base.get(name) or (jump[1] if jump and jump[0] == name else None)
    st.caption(f"종목코드: {code}" + ("  · 🧭 스크리너에서 선택한 종목" if jump and jump[0] == name else ""))

    with st.spinner(f"{name}({code}) 데이터를 불러오는 중..."):
        df = prepare(code, years, settings)

    if df is None:
        st.error("데이터를 가져오지 못했거나 선택한 기간에 표시할 데이터가 없습니다.")
        st.stop()

    result = run_backtest(
        df, stop_loss_pct=settings.stop_loss_pct, trailing_stop_pct=settings.trailing_stop_pct,
        exit_on_signal=settings.exit_on_signal, adaptive_exit=settings.adaptive_exit,
    )
    bh_return = buy_and_hold_return_pct(df)

    latest = df.iloc[-1]
    signal_color = {"매수": "green", "매도": "red", "관망": "gray"}[latest["SIGNAL"]]
    st.subheader(f"{name} ({code}) — 현재 시점 신호")
    st.markdown(
        f"### :{signal_color}[● {latest['SIGNAL']}]  "
        f"(점수 {latest['SCORE_TOTAL']:+d}, 기준일 {latest.name.date()})"
    )
    if settings.volume_filter_mode == "auto":
        regime_label = "상승장" if bool(latest["IS_BULL_REGIME"]) else "하락장"
        regime_color = "green" if bool(latest["IS_BULL_REGIME"]) else "red"
        vol_note = "" if bool(latest["IS_BULL_REGIME"]) else " → 거래량 필터 적용 중"
        st.caption(
            f"자동 거래량 필터: 현재 장세 :{regime_color}[**{regime_label}**]"
            f"(종가 vs 200일선){vol_note}"
        )
    render_order_link(name, str(code))

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("전략 누적수익률", f"{result.total_return_pct:+.1%}")
    c2.metric("단순보유 수익률", f"{bh_return:+.1%}")
    c3.metric("최대낙폭(MDD)", f"{result.mdd_pct:.1%}")
    c4.metric("총 매매 횟수", f"{result.num_trades}회")
    c5.metric("승률", f"{result.win_rate_pct:.0%}" if result.win_rate_pct is not None else "—")
    c6.metric("손절 청산", f"{result.num_stopped_out}회")
    c7.metric("트레일링 청산", f"{result.num_trailing_stopped_out}회")

    d1, d2, d3 = st.columns(3)
    d1.metric(
        "샤프 비율", f"{result.sharpe_ratio:.2f}" if result.sharpe_ratio is not None else "—",
        help="수익 대비 변동성(위험)을 감안한 지표. 높을수록 같은 위험 대비 수익이 좋다는 뜻.",
    )
    d2.metric(
        "소르티노 비율", f"{result.sortino_ratio:.2f}" if result.sortino_ratio is not None else "—",
        help="샤프 비율과 비슷하지만 '하락 변동성'만 위험으로 계산. 트레일링 스탑처럼 "
        "상승은 그냥 두고 하락만 방어하는 전략에서 샤프보다 더 적합할 수 있음.",
    )
    d3.metric(
        "시장 노출도", f"{result.exposure_pct:.0%}" if result.exposure_pct is not None else "—",
        help="전체 기간 중 실제로 주식을 들고 있었던 비율. 낮을수록 현금으로 대기한 "
        "기간이 길었다는 뜻 — 같은 수익률이면 노출도가 낮을수록 더 효율적이었다고 볼 수 있음.",
    )

    if result.num_trades > 0:
        total_days = (df.index.max() - df.index.min()).days
        avg_gap = total_days / result.num_trades
        st.caption(
            f"평균 보유일수 약 {result.avg_holding_days:.0f}일 · "
            f"평균 매매 주기 약 {avg_gap:.0f}일에 1회(매수+매도 한 쌍 기준)"
        )

    if result.total_return_pct < bh_return:
        st.info(
            "ℹ️ 이 기간 동안은 단순 보유(Buy & Hold)가 전략보다 수익률이 높았습니다. "
            "강한 상승장에서는 중간에 매도 신호로 빠져나오는 전략이 상승분을 다 못 먹는 경우가 많습니다."
        )

    chart_tf_label = st.radio(
        "차트 단위", list(CHART_TIMEFRAMES.keys()), index=0, horizontal=True,
        help="캔들차트를 일/주/월/년 단위로 바꿔서 볼 수 있습니다. 매수/매도 신호와 "
        "백테스트는 항상 일봉 기준 그대로이며, 이건 보기 편하라고 캔들만 다시 묶은 "
        "것입니다. 이평선/RSI/MACD도 선택한 단위 기준으로 다시 계산됩니다 (예: 주봉의 "
        "MA20은 20주 평균). 월봉·년봉은 조회 기간과 무관하게 더 오래된 데이터까지 "
        "따로 받아와서 캔들이 너무 적게 보이지 않도록 합니다. 다만 년봉은 '20년 "
        "이동평균' 같은 게 더 이상 의미가 없어서 이평선·RSI·MACD는 생략하고 캔들만 "
        "보여줍니다.",
    )
    # 2026-09-18: 월봉/년봉은 사이드바 조회 기간만으로는 캔들이 몇 개 안 남아서
    # 그래프가 휑해 보이는 문제가 있었다 — chart_dataframe()이 이 경우 신호/
    # 백테스트와 무관하게 차트 표시용으로만 더 오래된 데이터를 따로 받아온다.
    rule = CHART_TIMEFRAMES[chart_tf_label]
    chart_df = chart_dataframe(code, rule, df, years)
    if len(chart_df) < 5:
        st.warning(
            f"{chart_tf_label} 기준으로 표시할 캔들이 너무 적습니다 "
            "(이 종목의 상장 기간 자체가 짧을 수 있습니다)."
        )

    # 년봉에서는 이평선/RSI/MACD를 생략한다 — MA60을 년봉에 그대로 적용하면
    # "60년 이동평균"이 되어버리는데, 그만큼의 데이터도 없을뿐더러 있다 해도
    # 의미 있는 지표가 아니다(일/주/월봉과 달리 년봉은 지표 기간을 그대로
    # 늘려 쓸 수 있는 단위가 아님). 캔들 자체는 그대로 넓은 기간을 보여준다.
    show_indicators = rule != "YE"

    if show_indicators:
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[0.55, 0.2, 0.25],
            subplot_titles=(f"가격 · 이평선 · 볼린저밴드 ({chart_tf_label})", "RSI(14)", "MACD"),
        )
    else:
        fig = make_subplots(rows=1, cols=1, subplot_titles=(f"가격 ({chart_tf_label})",))

    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue",
    ), row=1, col=1)

    if show_indicators:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA_SHORT"], name=f"MA{ind.MA_SHORT}",
                                  line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MA_LONG"], name=f"MA{ind.MA_LONG}",
                                  line=dict(width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_UPPER"], name="BB상단",
                                  line=dict(width=1, dash="dot", color="gray")), row=1, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["BB_LOWER"], name="BB하단",
                                  line=dict(width=1, dash="dot", color="gray")), row=1, col=1)

    # 매수/매도 신호는 항상 일봉(daily) 기준 원본 df에서 가져온다 (신호는 리샘플링하지 않음)
    buys = df[df["SIGNAL"] == "매수"]
    sells = df[df["SIGNAL"] == "매도"]
    fig.add_trace(go.Scatter(x=buys.index, y=buys["Low"] * 0.97, mode="markers",
                              marker=dict(symbol="triangle-up", size=11, color="red"),
                              name="매수 신호"), row=1, col=1)
    fig.add_trace(go.Scatter(x=sells.index, y=sells["High"] * 1.03, mode="markers",
                              marker=dict(symbol="triangle-down", size=11, color="blue"),
                              name="매도 신호"), row=1, col=1)

    if show_indicators:
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], name="RSI", line=dict(color="purple")),
                      row=2, col=1)
        fig.add_hline(y=ind.RSI_OVERBOUGHT, line_dash="dot", line_color="red", row=2, col=1)
        fig.add_hline(y=ind.RSI_OVERSOLD, line_dash="dot", line_color="blue", row=2, col=1)

        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"], name="MACD", line=dict(color="orange")),
                      row=3, col=1)
        fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_SIGNAL"], name="Signal",
                                  line=dict(color="gray")), row=3, col=1)

    fig.update_layout(height=850 if show_indicators else 500, xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)
    if not show_indicators:
        st.caption(
            "ℹ️ 년봉에서는 이동평균·RSI·MACD를 표시하지 않습니다 — 예를 들어 "
            "MA60을 년봉에 그대로 적용하면 '60년 이동평균'이 되어버려서 데이터도 "
            "부족하고 의미도 없기 때문입니다. 지표를 보려면 일/주/월봉을 이용하세요."
        )

    with st.expander("최근 신호 로그"):
        log = df[df["SIGNAL"] != "관망"][
            ["Close", "SCORE_MA", "SCORE_RSI", "SCORE_MACD", "SCORE_BB", "SCORE_TOTAL", "SIGNAL"]
        ].sort_index(ascending=False)
        st.dataframe(log, use_container_width=True)

    # 2026-09-15 수정: 예전엔 "오늘 날짜"를 그냥 찍었는데, 주말/공휴일이거나
    # 데이터 출처가 아직 그날 시세를 안 올렸으면 실제로 화면에 쓰인 값은 그보다
    # 며칠 전 데이터인데도 "오늘 기준"이라고 표시돼서 혼동을 줬다. 실제로 df에
    # 들어있는 마지막 거래일을 보여주도록 고쳤다.
    st.caption(f"데이터 기준: {df.index.max().date()} · 출처: FinanceDataReader (KRX)")
