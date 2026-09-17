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

from core import (
    PERIOD_OPTIONS,
    VIEW_HOLDINGS,
    VIEW_OPS_NOTES,
    VIEW_SCREENER,
    VIEW_SINGLE_STOCK,
    VIEWS,
    Settings,
)
from data.fetch import load_settings, load_watchlist, save_settings
from views import holdings, ops_notes, screener, single_stock

st.set_page_config(page_title="주식 매매 타이밍 분석", layout="wide")

# 매 실행(rerun)마다 최신 관심종목을 다시 읽는다 (교체 적용 직후에도 바로 반영되도록)
WATCHLIST = load_watchlist()

st.title("📈 국내 주식 매수/매도 타이밍 분석")


def _safe_index(options: list[str], value, default: int) -> int:
    """저장된 값이 지금 옵션 목록에 없으면(예전 버전 흔적 등) 기본 인덱스로."""
    return options.index(value) if value in options else default


def _clamp(value, lo, hi):
    try:
        return max(lo, min(hi, value))
    except TypeError:
        return lo


# ---- 공통 설정 (사이드바) ----
# 2026-09-18: 이 설정들을 관심종목/보유종목과 같은 방식(Gist 또는 로컬 파일)
# 으로 저장해서, 재배포·새로고침 이후에도 마지막으로 골라둔 값이 그대로
# 남도록 했다 — 사용자가 직접 바꾸기 전까지는 유지된다.
_saved_settings = load_settings()

with st.sidebar:
    st.header("설정")
    period_options = list(PERIOD_OPTIONS.keys())
    period_label = st.selectbox(
        "조회 기간", period_options,
        index=_safe_index(period_options, _saved_settings["period_label"], 1),
    )
    years = PERIOD_OPTIONS[period_label]

    st.divider()
    st.subheader("신호 임계값")
    threshold = st.slider(
        "임계값 (낮을수록 신호가 자주 뜸)",
        min_value=1, max_value=3, value=_clamp(_saved_settings["threshold"], 1, 3),
        help="지표 4개(이평선/RSI/MACD/볼린저밴드) 중 몇 개 이상 동시에 같은 방향을 "
        "가리켜야 신호로 인정할지 결정합니다.",
    )

    st.divider()
    st.subheader("거래량 필터")
    volume_options = ["끔", "항상 켬", "자동(하락장에만)"]
    volume_mode_label = st.radio(
        "적용 방식", volume_options,
        index=_safe_index(volume_options, _saved_settings["volume_mode_label"], 2),
        help="신호가 뜬 날 거래량이 최근 평균보다 충분히 커야(관심이 실린 움직임) "
        "신호로 인정합니다. '자동'은 종가가 200일 이동평균 아래(하락장)일 때만 "
        "거래량 확인을 요구합니다 — 실측 결과 거래량 필터는 하락장 방어에는 "
        "거의 항상 도움이 되지만 상승장에서는 수익을 크게 깎는 경우가 많았습니다.",
    )
    volume_filter_mode: bool | str = {
        "끔": False, "항상 켬": True, "자동(하락장에만)": "auto",
    }[volume_mode_label]
    volume_multiplier = st.slider(
        "거래량 배수 (최근 20일 평균 대비)", min_value=1.0, max_value=2.0,
        value=_clamp(_saved_settings["volume_multiplier"], 1.0, 2.0), step=0.1,
        disabled=volume_mode_label == "끔",
    )

    st.divider()
    st.subheader("손절 규칙")
    use_stop_loss = st.checkbox(
        "손절 사용", value=bool(_saved_settings["use_stop_loss"]),
        help="매수가 대비 일정 % 하락하면 신호와 무관하게 즉시 청산합니다.",
    )
    stop_loss_slider = st.slider(
        "손절 기준(%)", min_value=3, max_value=20,
        value=_clamp(_saved_settings["stop_loss_slider"], 3, 20), step=1,
        disabled=not use_stop_loss,
    )
    stop_loss_pct = stop_loss_slider / 100 if use_stop_loss else None

    st.divider()
    st.subheader("청산(매도) 방식")
    exit_options = ["신호 기반 (기본)", "트레일링 스탑 단독", "적응형(상승장 트레일링+하락장 신호)"]
    exit_mode_label = st.radio(
        "어떻게 팔지 결정할지", exit_options,
        index=_safe_index(exit_options, _saved_settings["exit_mode_label"], 0),
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
    trailing_stop_slider = _saved_settings["trailing_stop_slider"]
    if exit_mode_label != "신호 기반 (기본)":
        trailing_stop_slider = st.slider(
            "트레일링 스탑 기준(%, 최고가 대비)", min_value=5, max_value=30,
            value=_clamp(_saved_settings["trailing_stop_slider"], 5, 30), step=1,
        )
        trailing_stop_pct = trailing_stop_slider / 100
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

_current_settings = {
    "period_label": period_label,
    "threshold": threshold,
    "volume_mode_label": volume_mode_label,
    "volume_multiplier": volume_multiplier,
    "use_stop_loss": use_stop_loss,
    "stop_loss_slider": stop_loss_slider,
    "exit_mode_label": exit_mode_label,
    "trailing_stop_slider": trailing_stop_slider,
}
if _current_settings != _saved_settings:
    save_settings(_current_settings)

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

if active_view == VIEW_SINGLE_STOCK:
    single_stock.render(WATCHLIST, years, settings)
elif active_view == VIEW_SCREENER:
    screener.render(WATCHLIST, years, settings)
elif active_view == VIEW_HOLDINGS:
    holdings.render(years, settings)
elif active_view == VIEW_OPS_NOTES:
    ops_notes.render()
