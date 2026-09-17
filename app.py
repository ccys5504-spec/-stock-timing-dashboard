"""국내 주식 매수/매도 타이밍 분석 대시보드 (Streamlit) — 진입점.

각 탭(뷰)의 실제 화면 내용은 views/ 패키지에 나눠져 있다. 여기서는
(1) 페이지 설정, (2) 사이드바에서 공통 설정값 받기, (3) 선택된 탭에 맞는
views.*.render()를 호출하는 역할만 한다.

2026-09-15 정리: 예전에는 이 파일 하나에 탭 5개 내용이 전부(800줄 가까이)
들어있어서 원하는 부분을 찾기 어려웠다. 화면에 보이는 내용/동작은 이 정리
전후로 달라진 게 없다 — 순수하게 코드 위치만 옮겼다.
"""
from __future__ import annotations

import _pyarrow_compat  # noqa: F401  # pandas/streamlit보다 반드시 먼저 임포트

import streamlit as st

from core import PERIOD_OPTIONS, VIEWS, Settings
from data.fetch import load_watchlist
from views import holdings, ops_notes, screener, single_stock

st.set_page_config(page_title="주식 매매 타이밍 분석", layout="wide")

# 매 실행(rerun)마다 최신 관심종목을 다시 읽는다 (교체 적용 직후에도 바로 반영되도록)
WATCHLIST = load_watchlist()

st.title("📈 국내 주식 매수/매도 타이밍 분석")

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

settings = Settings(
    threshold=threshold,
    volume_filter_mode=volume_filter_mode,
    volume_multiplier=volume_multiplier,
    stop_loss_pct=stop_loss_pct,
    trailing_stop_pct=trailing_stop_pct,
    exit_on_signal=exit_on_signal,
    adaptive_exit=adaptive_exit,
)

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

if active_view == VIEWS[0]:
    single_stock.render(WATCHLIST, years, settings)
elif active_view == VIEWS[1]:
    screener.render(WATCHLIST, years, settings)
elif active_view == VIEWS[2]:
    holdings.render(years, settings)
else:
    ops_notes.render()
