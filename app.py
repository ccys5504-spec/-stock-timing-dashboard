"""국내 주식 매수/매도 타이밍 분석 대시보드 (Streamlit)."""
from __future__ import annotations

import datetime as dt
import io

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from backtest.engine import buy_and_hold_return_pct, run_backtest
from data.fetch import (
    fetch_ohlcv,
    get_universe,
    load_holdings,
    load_watchlist,
    resolve_stock_name,
    save_holdings,
    save_watchlist,
)
from screener.scan import scan_signals
from signals import indicators as ind

st.set_page_config(page_title="주식 매매 타이밍 분석", layout="wide")

PERIOD_OPTIONS = {"1년": 1, "3년": 3, "5년": 5}
VIEWS = ["🔍 단일 종목 분석", "📊 3종목 비교", "🧭 종목 추천(스크리너)", "💼 내 보유종목"]

# 매 실행(rerun)마다 최신 관심종목을 다시 읽는다 (교체 적용 직후에도 바로 반영되도록)
WATCHLIST = load_watchlist()

st.title("📈 국내 주식 매수/매도 타이밍 분석")
st.warning(
    "⚠️ 이 대시보드는 **투자 참고용**이며 투자 자문이 아닙니다. "
    "과거 성과가 미래 수익을 보장하지 않으며, 실제 매매 판단과 책임은 본인에게 있습니다. "
    "이 프로그램은 실제 주문을 실행하지 않습니다."
)

# ---- 공통 설정 (사이드바) ----
with st.sidebar:
    st.header("설정")
    period_label = st.selectbox("조회 기간", list(PERIOD_OPTIONS.keys()), index=1)
    years = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("신호 임계값")
    threshold = st.slider(
        "임계값 (낮을수록 신호가 자주 뜸)",
        min_value=1, max_value=3, value=1,
        help="지표 4개(이평선/RSI/MACD/볼린저밴드) 중 몇 개 이상 동시에 같은 방향을 "
        "가리켜야 신호로 인정할지 결정합니다.",
    )

    st.divider()
    st.subheader("거래량 필터")
    volume_mode_label = st.radio(
        "적용 방식",
        ["끔", "항상 켬", "자동(하락장에만)"],
        index=2,
        help="신호가 뜬 날 거래량이 최근 평균보다 충분히 커야(관심이 실린 움직임) "
        "신호로 인정합니다. '자동'은 종가가 200일 이동평균 아래(하락장)일 때만 "
        "거래량 확인을 요구합니다 — 실측 결과 거래량 필터는 하락장 방어에는 "
        "거의 항상 도움이 되지만 상승장에서는 수익을 크게 깎는 경우가 많았습니다.",
    )
    volume_filter_mode: bool | str = {
        "끔": False, "항상 켬": True, "자동(하락장에만)": "auto",
    }[volume_mode_label]
    volume_multiplier = st.slider(
        "거래량 배수 (최근 20일 평균 대비)", min_value=1.0, max_value=2.0, value=1.2, step=0.1,
        disabled=volume_mode_label == "끔",
    )

    st.divider()
    st.subheader("손절 규칙")
    use_stop_loss = st.checkbox(
        "손절 사용", value=False,
        help="매수가 대비 일정 % 하락하면 신호와 무관하게 즉시 청산합니다.",
    )
    stop_loss_pct = st.slider(
        "손절 기준(%)", min_value=3, max_value=20, value=10, step=1,
        disabled=not use_stop_loss,
    ) / 100 if use_stop_loss else None

    st.divider()
    st.subheader("청산(매도) 방식")
    exit_mode_label = st.radio(
        "어떻게 팔지 결정할지",
        ["신호 기반 (기본)", "트레일링 스탑 단독", "적응형(상승장 트레일링+하락장 신호)"],
        index=0,
        help="'신호 기반'은 매도 신호가 뜨면 바로 팝니다(하락장 방어에 유리, "
        "다만 강한 상승장에서는 일찍 팔아 수익을 못 챙길 수 있음). '트레일링 스탑 "
        "단독'은 매도 신호를 무시하고 보유 중 최고가 대비 일정 % 하락할 때만 "
        "팝니다(상승장에서 수익을 훨씬 키우지만, 하락장에서는 손실도 커짐). "
        "'적응형'은 상승장엔 트레일링, 하락장엔 신호 기반으로 자동 전환합니다 "
        "(둘의 중간 정도 성격 — 어느 한쪽보다 항상 낫지는 않습니다).",
    )
    trailing_stop_pct = None
    exit_on_signal = True
    adaptive_exit = False
    if exit_mode_label != "신호 기반 (기본)":
        trailing_stop_pct = st.slider(
            "트레일링 스탑 기준(%, 최고가 대비)", min_value=5, max_value=30, value=20, step=1,
        ) / 100
        if exit_mode_label == "트레일링 스탑 단독":
            exit_on_signal = False
        else:
            adaptive_exit = True
    st.caption(
        "⚠️ 실측 결과(README): 트레일링 스탑 단독은 최근 5년 상승장에서 수익률을 "
        "몇 배로 키웠지만, 2022년 하락장에서는 신호 기반보다 손실이 더 컸습니다. "
        "정답은 없고 '수익 극대화 vs 손실 방어' 중 어디에 무게를 둘지의 선택입니다."
    )

    st.divider()
    st.caption("💾 대시보드에서 만든 CSV/엑셀 파일을 구글드라이브 등에 옮기고 싶다면, "
               "다운로드한 뒤 직접 업로드하거나 Claude에게 'OO 파일 구글드라이브에 저장해줘'라고 "
               "요청하면 됩니다.")


@st.cache_data(ttl=3600, show_spinner=False)
def load_data(code: str, years: int) -> pd.DataFrame:
    return fetch_ohlcv(code, years=years)


def prepare(code: str, years: int) -> pd.DataFrame | None:
    """데이터 로드 + 지표/신호 계산 + 기간 컷까지 한 번에."""
    try:
        raw = load_data(code, years)
    except Exception:  # noqa: BLE001
        return None
    df = ind.build_signals(
        raw,
        buy_threshold=threshold,
        sell_threshold=-threshold,
        volume_filter=volume_filter_mode,
        volume_multiplier=volume_multiplier,
    )
    df = df.dropna(subset=["MA_LONG"])
    cutoff = df.index.max() - pd.Timedelta(days=int(years * 365.25))
    df = df[df.index >= cutoff]
    return df if not df.empty else None


@st.cache_data(ttl=1800, show_spinner=False)
def run_screener(markets, top_n, threshold_, vfilter, vmult):
    universe = get_universe(markets=tuple(markets), top_n=top_n)
    return scan_signals(
        universe, years=1,
        buy_threshold=threshold_, sell_threshold=-threshold_,
        volume_filter=vfilter, volume_multiplier=vmult,
    )


CHART_TIMEFRAMES = {"일봉": None, "주봉": "W", "월봉": "ME", "년봉": "YE"}


def resample_for_chart(df: pd.DataFrame, rule: str | None) -> pd.DataFrame:
    """일봉 df를 주봉/월봉/년봉으로 리샘플링하고, 그 기준으로 지표를 다시 계산한다
    (차트 표시 전용 — 매매 신호/백테스트는 항상 일봉 기준 그대로 유지된다).
    """
    if rule is None:
        return df
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    resampled = df[list(agg.keys())].resample(rule).agg(agg).dropna(subset=["Close"])
    return ind.add_indicators(resampled)


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # BOM 포함 -> 엑셀에서 한글 깨짐 방지


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="스캔결과")
    return buf.getvalue()


# ---- 뷰(탭) 상태: 스크리너에서 종목을 클릭하면 여기로 전환된다 ----
# 주의: st.radio가 만들어진 '이후'에는 그 key(active_view)를 직접 수정할 수 없다
# (StreamlitWidgetAlreadyInstantiatedError). 그래서 전환 요청은 별도 키
# (_pending_view)에 적어두고, 위젯을 만들기 '전'에 반영한다.
if "active_view" not in st.session_state:
    st.session_state["active_view"] = VIEWS[0]
if "jump" not in st.session_state:
    st.session_state["jump"] = None  # (name, code) 튜플 또는 None
if st.session_state.get("_pending_view"):
    st.session_state["active_view"] = st.session_state.pop("_pending_view")

active_view = st.radio(
    "보기 선택", VIEWS, key="active_view", horizontal=True, label_visibility="collapsed"
)

# =====================================================================
# 뷰 1: 단일 종목 분석
# =====================================================================
if active_view == VIEWS[0]:
    jump = st.session_state["jump"]
    options = list(WATCHLIST.keys())
    if jump and jump[0] not in options:
        options = [jump[0]] + options
    default_index = options.index(jump[0]) if jump and jump[0] in options else 0

    name = st.selectbox("종목 선택 (관심종목 + 스크리너에서 넘어온 종목)", options, index=default_index)
    code = WATCHLIST.get(name) or (jump[1] if jump and jump[0] == name else None)
    st.caption(f"종목코드: {code}" + ("  · 🧭 스크리너에서 선택한 종목" if jump and jump[0] == name else ""))

    with st.spinner(f"{name}({code}) 데이터를 불러오는 중..."):
        df = prepare(code, years)

    if df is None:
        st.error("데이터를 가져오지 못했거나 선택한 기간에 표시할 데이터가 없습니다.")
        st.stop()

    result = run_backtest(
        df, stop_loss_pct=stop_loss_pct, trailing_stop_pct=trailing_stop_pct,
        exit_on_signal=exit_on_signal, adaptive_exit=adaptive_exit,
    )
    bh_return = buy_and_hold_return_pct(df)

    latest = df.iloc[-1]
    signal_color = {"매수": "green", "매도": "red", "관망": "gray"}[latest["SIGNAL"]]
    st.subheader(f"{name} ({code}) — 현재 시점 신호")
    st.markdown(
        f"### :{signal_color}[● {latest['SIGNAL']}]  "
        f"(점수 {latest['SCORE_TOTAL']:+d}, 기준일 {latest.name.date()})"
    )
    if volume_filter_mode == "auto":
        regime_label = "상승장" if bool(latest["IS_BULL_REGIME"]) else "하락장"
        regime_color = "green" if bool(latest["IS_BULL_REGIME"]) else "red"
        vol_note = "" if bool(latest["IS_BULL_REGIME"]) else " → 거래량 필터 적용 중"
        st.caption(
            f"자동 거래량 필터: 현재 장세 :{regime_color}[**{regime_label}**]"
            f"(종가 vs 200일선){vol_note}"
        )

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("전략 누적수익률", f"{result.total_return_pct:+.1%}")
    c2.metric("단순보유 수익률", f"{bh_return:+.1%}")
    c3.metric("최대낙폭(MDD)", f"{result.mdd_pct:.1%}")
    c4.metric("총 매매 횟수", f"{result.num_trades}회")
    c5.metric("승률", f"{result.win_rate_pct:.0%}" if result.win_rate_pct is not None else "—")
    c6.metric("손절 청산", f"{result.num_stopped_out}회")
    c7.metric("트레일링 청산", f"{result.num_trailing_stopped_out}회")

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
        "MA20은 20주 평균). 월봉·년봉은 조회 기간이 짧으면 이평선이 계산에 필요한 "
        "구간이 부족해 일부 표시되지 않을 수 있습니다.",
    )
    chart_df = resample_for_chart(df, CHART_TIMEFRAMES[chart_tf_label])
    if len(chart_df) < 5:
        st.warning(
            f"{chart_tf_label} 기준으로 표시할 캔들이 너무 적습니다 "
            "(사이드바 '조회 기간'을 늘려보세요)."
        )

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        row_heights=[0.55, 0.2, 0.25],
        subplot_titles=(f"가격 · 이평선 · 볼린저밴드 ({chart_tf_label})", "RSI(14)", "MACD"),
    )

    fig.add_trace(go.Candlestick(
        x=chart_df.index, open=chart_df["Open"], high=chart_df["High"],
        low=chart_df["Low"], close=chart_df["Close"],
        name="가격", increasing_line_color="red", decreasing_line_color="blue",
    ), row=1, col=1)
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

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["RSI"], name="RSI", line=dict(color="purple")),
                  row=2, col=1)
    fig.add_hline(y=ind.RSI_OVERBOUGHT, line_dash="dot", line_color="red", row=2, col=1)
    fig.add_hline(y=ind.RSI_OVERSOLD, line_dash="dot", line_color="blue", row=2, col=1)

    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD"], name="MACD", line=dict(color="orange")),
                  row=3, col=1)
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df["MACD_SIGNAL"], name="Signal",
                              line=dict(color="gray")), row=3, col=1)

    fig.update_layout(height=850, xaxis_rangeslider_visible=False,
                       legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("최근 신호 로그"):
        log = df[df["SIGNAL"] != "관망"][
            ["Close", "SCORE_MA", "SCORE_RSI", "SCORE_MACD", "SCORE_BB", "SCORE_TOTAL", "SIGNAL"]
        ].sort_index(ascending=False)
        st.dataframe(log, use_container_width=True)

    st.caption(f"데이터 기준: {dt.date.today()} · 출처: FinanceDataReader (KRX)")

# =====================================================================
# 뷰 2: 3종목 비교
# =====================================================================
elif active_view == VIEWS[1]:
    st.caption("사이드바에서 설정한 임계값/거래량 필터/손절 규칙이 아래 종목에 동일하게 적용됩니다.")

    rows = []
    equity_curves = {}
    signal_badges = {}

    with st.spinner("종목별 데이터를 불러오고 백테스트하는 중..."):
        for name, code in WATCHLIST.items():
            df = prepare(code, years)
            if df is None:
                rows.append({"종목": name, "오류": "데이터 없음"})
                continue

            result = run_backtest(
                df, stop_loss_pct=stop_loss_pct, trailing_stop_pct=trailing_stop_pct,
                exit_on_signal=exit_on_signal, adaptive_exit=adaptive_exit,
            )
            bh_return = buy_and_hold_return_pct(df)
            latest = df.iloc[-1]

            rows.append({
                "종목": name,
                "현재 신호": latest["SIGNAL"],
                "전략 수익률": result.total_return_pct,
                "단순보유 수익률": bh_return,
                "MDD": result.mdd_pct,
                "매매횟수": result.num_trades,
                "승률": result.win_rate_pct,
            })
            equity_curves[name] = result.equity_curve / result.equity_curve.iloc[0]
            signal_badges[name] = latest["SIGNAL"]

    summary_df = pd.DataFrame(rows)

    cols = st.columns(len(WATCHLIST))
    for col, (name, _) in zip(cols, WATCHLIST.items()):
        sig = signal_badges.get(name, "—")
        color = {"매수": "green", "매도": "red", "관망": "gray"}.get(sig, "gray")
        with col:
            st.markdown(f"**{name}**")
            st.markdown(f"### :{color}[● {sig}]")

    st.divider()

    st.subheader("백테스트 요약 비교")
    display_df = summary_df.copy()
    for pct_col in ["전략 수익률", "단순보유 수익률", "MDD", "승률"]:
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].apply(
                lambda v: f"{v:+.1%}" if pd.notna(v) else "—"
            )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    best = summary_df.loc[summary_df["전략 수익률"].idxmax(), "종목"] if "전략 수익률" in summary_df else None
    if best:
        st.caption(f"💡 이 설정·기간 기준으로는 **{best}**의 전략 수익률이 가장 높았습니다.")

    st.subheader("전략 누적수익률 곡선 비교 (시작점 = 1.0)")
    fig2 = go.Figure()
    for name, curve in equity_curves.items():
        fig2.add_trace(go.Scatter(x=curve.index, y=curve.values, name=name, mode="lines"))
    fig2.update_layout(height=450, yaxis_title="누적 배수 (1.0 = 원금)",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "⚠️ 여러 조합을 시험해보며 '가장 수익률 높은 설정'을 찾는 것 자체가 과최적화(overfitting) "
        "위험이 있습니다. 과거 데이터 한 번에 잘 맞는 설정이 미래에도 맞는다는 보장은 없습니다 — "
        "상승장/하락장 등 서로 다른 기간에서 두루 검증(scripts/optimize.py)해보고 극단적으로 "
        "낙관적인 조합은 의심하는 게 안전합니다."
    )

# =====================================================================
# 뷰 3: 종목 추천 (스크리너)
# =====================================================================
elif active_view == VIEWS[2]:
    st.caption(
        "관심 종목 3개에 국한하지 않고, 시가총액 상위 종목들을 훑어서 "
        "**지금 시점에 매수/매도 신호가 뜬 종목**을 찾아줍니다."
    )
    st.info(
        "ℹ️ 아래 목록은 사이드바의 임계값·거래량 필터를 그대로 적용해 기계적으로 계산한 "
        "결과이며, **투자 자문이 아닙니다.** 시가총액 상위 종목 위주로만 스캔하므로 "
        "거래량이 적은 소형주는 포함되지 않습니다. 신호가 떴다고 바로 매매하지 말고, "
        "반드시 재무상태·뉴스 등을 직접 확인하세요."
    )

    col_a, col_b, col_c = st.columns([1.2, 1, 1])
    with col_a:
        scan_markets = st.multiselect(
            "대상 시장", ["KOSPI", "KOSDAQ"], default=["KOSPI", "KOSDAQ"],
        )
    with col_b:
        scan_top_n = st.slider("스캔할 종목 수 (시가총액 상위)", min_value=20, max_value=300,
                                value=100, step=10)
    with col_c:
        only_actionable = st.checkbox("매수/매도 신호만 보기", value=True)

    run_scan = st.button("🔎 스캔 시작", type="primary")

    if run_scan:
        if not scan_markets:
            st.error("대상 시장을 1개 이상 선택하세요.")
        else:
            with st.spinner(f"{scan_top_n}개 종목을 스캔하는 중... (수십 초 소요될 수 있습니다)"):
                scan_result = run_screener(
                    scan_markets, scan_top_n, threshold, volume_filter_mode, volume_multiplier
                )
            st.session_state["scan_result"] = scan_result

    scan_result = st.session_state.get("scan_result")
    if scan_result is None:
        st.caption("👆 '스캔 시작'을 누르면 결과가 여기에 표시됩니다.")
    elif scan_result.empty:
        st.warning("스캔 결과가 없습니다. 데이터 조회에 실패했을 수 있습니다.")
    else:
        n_buy = (scan_result["신호"] == "매수").sum()
        n_sell = (scan_result["신호"] == "매도").sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("스캔한 종목 수", len(scan_result))
        m2.metric("매수 신호", f"{n_buy}개")
        m3.metric("매도 신호", f"{n_sell}개")

        display = scan_result[scan_result["신호"] != "관망"] if only_actionable else scan_result
        display = display.reset_index(drop=True)

        def _highlight_signal(row):
            color = {"매수": "background-color: rgba(0,180,0,0.15)",
                     "매도": "background-color: rgba(200,0,0,0.12)"}.get(row["신호"], "")
            return [color] * len(row)

        st.caption("💡 표에서 종목 행을 클릭하면 '단일 종목 분석'으로 바로 이동해 상세 차트를 볼 수 있습니다.")
        event = st.dataframe(
            display.style.apply(_highlight_signal, axis=1),
            use_container_width=True, hide_index=True,
            on_select="rerun", selection_mode="single-row", key="scan_table",
        )
        selected_rows = []
        try:
            selected_rows = list(event.selection.rows)
        except Exception:  # noqa: BLE001
            try:
                selected_rows = list(event["selection"]["rows"])
            except Exception:  # noqa: BLE001
                selected_rows = []

        if selected_rows:
            picked = display.iloc[selected_rows[0]]
            st.session_state["jump"] = (picked["종목명"], picked["종목코드"])
            st.session_state["_pending_view"] = VIEWS[0]
            st.rerun()

        # ---- 파일로 저장 ----
        dl_col1, dl_col2 = st.columns(2)
        dl_col1.download_button(
            "⬇️ CSV로 다운로드", data=to_csv_bytes(display),
            file_name=f"stock_scan_{dt.date.today().isoformat()}.csv", mime="text/csv",
        )
        dl_col2.download_button(
            "⬇️ 엑셀로 다운로드", data=to_excel_bytes(display),
            file_name=f"stock_scan_{dt.date.today().isoformat()}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        st.caption(
            "점수는 이평선/RSI/MACD/볼린저밴드 4개 지표 합산값입니다."
        )

        # ---- 관심종목 교체 제안 ----
        st.divider()
        st.subheader("💡 관심종목 교체 제안")
        st.caption(
            "현재 관심종목 3개의 '지금 시점 점수'와, 스캔 결과 중 관심종목이 아닌 종목의 "
            "점수를 비교합니다. 이 점수는 백테스트 성과가 아니라 오늘 시점 신호의 강도일 "
            "뿐이니 참고 정도로만 활용하세요."
        )

        current_scores = {}
        for name, code in WATCHLIST.items():
            match = scan_result[scan_result["종목코드"] == code]
            if not match.empty:
                current_scores[name] = int(match.iloc[0]["점수"])
            else:
                fallback = prepare(code, years=1)
                current_scores[name] = int(fallback.iloc[-1]["SCORE_TOTAL"]) if fallback is not None else None

        candidates = scan_result[~scan_result["종목코드"].isin(WATCHLIST.values())]
        candidates = candidates.sort_values("점수", ascending=False)

        col_cur, col_cand = st.columns(2)
        with col_cur:
            st.markdown("**현재 관심종목**")
            cur_df = pd.DataFrame(
                [{"종목명": n, "점수": s if s is not None else "—"} for n, s in current_scores.items()]
            )
            st.dataframe(cur_df, use_container_width=True, hide_index=True)
        with col_cand:
            st.markdown("**교체 후보 (관심종목 제외, 스캔 점수 상위)**")
            st.dataframe(
                candidates[["종목명", "종목코드", "신호", "점수"]].head(5),
                use_container_width=True, hide_index=True,
            )

        valid_scores = {n: s for n, s in current_scores.items() if s is not None}
        if valid_scores and not candidates.empty:
            weakest_name = min(valid_scores, key=lambda k: valid_scores[k])
            weakest_score = valid_scores[weakest_name]
            best_candidate = candidates.iloc[0]

            if best_candidate["점수"] > weakest_score:
                st.info(
                    f"제안: 관심종목 중 점수가 가장 낮은 **{weakest_name}**(점수 {weakest_score:+d})를 "
                    f"**{best_candidate['종목명']}**(점수 {int(best_candidate['점수']):+d})로 "
                    "교체하는 걸 고려해볼 수 있습니다."
                )
                if st.button(f"✅ {weakest_name} → {best_candidate['종목명']} 교체 적용", type="primary"):
                    new_watchlist = {k: v for k, v in WATCHLIST.items() if k != weakest_name}
                    new_watchlist[best_candidate["종목명"]] = best_candidate["종목코드"]
                    save_watchlist(new_watchlist)
                    st.success(
                        f"관심종목을 교체했습니다: {weakest_name} → {best_candidate['종목명']}. "
                        "'3종목 비교' 탭에서 바로 확인해보세요."
                    )
                    st.rerun()
            else:
                st.caption("현재 관심종목이 스캔 후보들보다 점수가 더 높거나 같아서 교체를 제안하지 않습니다.")
        elif candidates.empty:
            st.caption("스캔 대상 중 관심종목을 제외한 후보가 없습니다 (스캔 범위를 넓혀보세요).")

# =====================================================================
# 뷰 4: 내 보유종목
# =====================================================================
else:
    st.caption(
        "실제로 보유 중인 종목의 수량·매입단가를 입력해두면, 현재가 대비 평가손익과 "
        "지금 시점 매수/매도 신호를 한 화면에서 확인할 수 있습니다."
    )
    st.info(
        "ℹ️ 여기 입력한 정보는 이 컴퓨터의 data/holdings.json 파일에만 저장됩니다. "
        "어디로도 전송되지 않습니다."
    )

    holdings = load_holdings()
    holdings_df = pd.DataFrame(holdings) if holdings else pd.DataFrame(
        columns=["종목코드", "수량", "매입단가"]
    )
    for col in ["종목코드", "수량", "매입단가"]:
        if col not in holdings_df.columns:
            holdings_df[col] = pd.Series(dtype="object")

    st.subheader("보유종목 입력")
    edited_df = st.data_editor(
        holdings_df[["종목코드", "수량", "매입단가"]],
        num_rows="dynamic",
        use_container_width=True,
        key="holdings_editor",
        column_config={
            "종목코드": st.column_config.TextColumn("종목코드", help="6자리 종목코드, 예: 000660", width="small"),
            "수량": st.column_config.NumberColumn("수량(주)", min_value=0, step=1),
            "매입단가": st.column_config.NumberColumn("매입단가(원)", min_value=0, step=100),
        },
    )

    if st.button("💾 저장", type="primary"):
        clean = edited_df.dropna(subset=["종목코드"])
        clean = clean[clean["종목코드"].astype(str).str.strip() != ""]
        save_holdings(clean.to_dict("records"))
        st.success("저장했습니다.")
        st.rerun()

    saved_holdings = load_holdings()
    if not saved_holdings:
        st.caption("아직 입력된 보유종목이 없습니다. 위 표에 입력 후 저장을 눌러주세요.")
        st.stop()

    st.divider()
    st.subheader("현재 상태")

    rows = []
    sell_alerts = []
    stoploss_alerts = []
    total_cost = 0.0
    total_value = 0.0

    with st.spinner("보유종목 현재가·신호를 확인하는 중..."):
        for h in saved_holdings:
            code = str(h.get("종목코드", "")).strip().zfill(6)
            qty = float(h.get("수량") or 0)
            buy_price = float(h.get("매입단가") or 0)
            if not code or qty <= 0 or buy_price <= 0:
                continue

            name = resolve_stock_name(code) or code
            df = prepare(code, years=1)
            if df is None:
                rows.append({"종목명": name, "종목코드": code, "오류": "데이터 조회 실패"})
                continue

            latest = df.iloc[-1]
            cur_price = float(latest["Close"])
            cost = buy_price * qty
            value = cur_price * qty
            pnl_pct = cur_price / buy_price - 1
            total_cost += cost
            total_value += value

            rows.append({
                "종목명": name,
                "종목코드": code,
                "수량": qty,
                "매입단가": buy_price,
                "현재가": cur_price,
                "평가손익률": pnl_pct,
                "평가손익(원)": value - cost,
                "현재 신호": latest["SIGNAL"],
            })

            if latest["SIGNAL"] == "매도":
                sell_alerts.append(name)
            if stop_loss_pct is not None and pnl_pct <= -stop_loss_pct:
                stoploss_alerts.append(f"{name} ({pnl_pct:+.1%})")

    if sell_alerts:
        st.warning(f"🔔 보유 중인 종목 중 **매도 신호**가 뜬 종목: {', '.join(sell_alerts)}")
    if stoploss_alerts:
        st.error(
            f"⚠️ 사이드바 손절 기준(-{stop_loss_pct:.0%})에 도달한 종목: {', '.join(stoploss_alerts)}"
        )

    if total_cost > 0:
        c1, c2, c3 = st.columns(3)
        c1.metric("총 매입금액", f"{total_cost:,.0f}원")
        c2.metric("총 평가금액", f"{total_value:,.0f}원")
        c3.metric("총 평가손익", f"{total_value - total_cost:+,.0f}원", f"{total_value/total_cost - 1:+.1%}")

    display_rows = pd.DataFrame(rows)
    if not display_rows.empty:
        fmt = display_rows.copy()
        if "평가손익률" in fmt.columns:
            fmt["평가손익률"] = fmt["평가손익률"].apply(lambda v: f"{v:+.1%}" if pd.notna(v) else "—")
        if "평가손익(원)" in fmt.columns:
            fmt["평가손익(원)"] = fmt["평가손익(원)"].apply(lambda v: f"{v:+,.0f}" if pd.notna(v) else "—")
        for c in ["매입단가", "현재가"]:
            if c in fmt.columns:
                fmt[c] = fmt[c].apply(lambda v: f"{v:,.0f}" if pd.notna(v) else "—")

        def _highlight(row):
            color = ""
            if row.get("현재 신호") == "매도":
                color = "background-color: rgba(200,0,0,0.12)"
            elif row.get("현재 신호") == "매수":
                color = "background-color: rgba(0,180,0,0.15)"
            return [color] * len(row)

        st.dataframe(fmt.style.apply(_highlight, axis=1), use_container_width=True, hide_index=True)

    st.caption(
        "⚠️ 여기 표시되는 신호는 사이드바에 설정된 임계값/거래량 필터/손절 기준을 그대로 "
        "적용한 기계적 계산 결과이며, 투자 자문이 아닙니다."
    )
