"""뷰 3: 종목 추천 (스크리너).

2026-09-29: 예전에는 "⭐ 오늘의 추천"(시총 상위 100개 자동 스캔)과 그 아래 "직접 범위를 정해서 스캔하기"
(시장·종목 수 선택 + 스캔 시작 버튼)가 따로 있었다. 하나로 합쳐서, 탭을 열면 시가총액 상위 300개(KOSPI+KOSDAQ)를
자동으로 스캔해 추천 종목을 고르고, 같은 스캔 결과로 전체 표·다운로드·관심종목 교체 제안까지 보여준다.
"""
from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import streamlit as st

from core import VIEW_SINGLE_STOCK, Settings, prepare, run_screener, to_csv_bytes, to_excel_bytes
from data.fetch import save_watchlist
from views.order_link import render_order_link

logger = logging.getLogger(__name__)

TODAY_PICK_UNIVERSE = 300  # "오늘의 추천"이 훑어보는 시가총액 상위 종목 수 (고정, KOSPI+KOSDAQ)
FAILURE_WARN_RATIO = 0.05  # 실패 종목이 이 비율 이상이면 눈에 띄는 경고로 표시


def _render_scan_health(scan_result: pd.DataFrame) -> None:
    """스캔이 실제로 몇 종목을 성공/실패했고 데이터가 언제 기준인지 보여준다.

    예전에는 일부 종목이 조용히 빠져도 "정상 결과"처럼 보였다(2026-09-20 점검서
    지적). 요청 수·성공 수·실패 종목·데이터 기준일(다른 종목보다 오래된 종목
    수 포함)을 항상 표시한다.
    """
    requested = scan_result.attrs.get("requested")
    failed = scan_result.attrs.get("failed", [])
    if requested is None:
        return

    ok = requested - len(failed)
    parts = [f"조회 성공 {ok}/{requested}개"]
    if failed:
        parts.append(f"실패 {len(failed)}개")
    if "기준일" in scan_result.columns and not scan_result.empty:
        latest = scan_result["기준일"].max()
        stale = int((scan_result["기준일"] < latest).sum())
        parts.append(f"데이터 기준일 {latest}" + (f"(이보다 오래된 종목 {stale}개)" if stale else ""))
    message = " · ".join(parts)

    if failed and requested and len(failed) / requested >= FAILURE_WARN_RATIO:
        st.warning(f"⚠️ {message} — 실패 비율이 높아 결과가 불완전할 수 있습니다.")
    else:
        st.caption(message)
    if failed:
        with st.expander(f"조회 실패한 {len(failed)}종목과 사유 보기"):
            st.dataframe(
                pd.DataFrame(failed, columns=["종목코드", "종목명", "사유"]),
                use_container_width=True, hide_index=True,
            )


def _run_today_scan(settings: Settings) -> pd.DataFrame | None:
    """시가총액 상위 300개를 사이드바 설정 기준으로 스캔한다. run_screener가 30분 캐싱하므로 같은 설정으로
    여러 번 열어봐도 매번 새로 스캔하지 않는다. 실패하면 화면에 알리고 None."""
    with st.spinner(f"시가총액 상위 {TODAY_PICK_UNIVERSE}개 종목을 훑어보는 중... (처음엔 1~2분쯤 걸릴 수 있습니다)"):
        try:
            return run_screener(
                ["KOSPI", "KOSDAQ"], TODAY_PICK_UNIVERSE,
                settings.threshold, settings.volume_filter_mode, settings.volume_multiplier,
            )
        except Exception as e:  # noqa: BLE001 — 종목별 실패가 아니라 유니버스 조회(KRX 전체 목록) 자체가
            # 막혔을 때 여기로 떨어진다. 예전에는 "데이터 조회 실패"로만 뭉뚱그려서 원인을 알 수 없었다
            # (2026-10-01, Cloud에서 실제로 겪음) — 이제 원인을 화면과 로그에 함께 남긴다.
            logger.warning("오늘의 추천 스캔 실패: %s: %s", type(e).__name__, e)
            st.warning(
                f"오늘의 추천을 계산하지 못했습니다(데이터 조회 실패: {type(e).__name__}: {e}). "
                "보통 일시적인 네트워크 문제이니 '다시 스캔'을 눌러보세요."
            )
            _render_rescan_button()
            return None


def _render_rescan_button() -> None:
    if st.button("🔄 다시 스캔", key="rescan", help="30분 동안 저장해 둔 스캔 결과를 버리고 새로 조회합니다."):
        run_screener.clear()
        st.rerun()


def _render_today_pick(scan_result: pd.DataFrame, settings: Settings) -> None:
    """스캔 결과에서 점수가 가장 높은 매수 후보 1개를 카드로 보여준다 — 매일 한 번 열어보는 용도."""
    st.markdown("### ⭐ 오늘의 추천")
    _render_scan_health(scan_result)
    if scan_result.empty:
        st.caption("스캔 결과가 없습니다.")
        _render_rescan_button()
        return

    buys = scan_result[scan_result["신호"] == "매수"]
    if buys.empty:
        st.info(
            f"오늘은 시가총액 상위 {TODAY_PICK_UNIVERSE}개 종목 중 매수 신호가 뜬 종목이 "
            "없습니다. 신호가 없으면 관망하는 것도 하나의 선택입니다."
        )
        _render_rescan_button()
        return

    pick = buys.iloc[0]  # run_screener가 이미 점수 내림차순으로 정렬해서 줌
    with st.container(border=True):
        st.markdown(f"#### 🟢 {pick['종목명']} ({pick['종목코드']}) · {pick['시장']}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가", f"{pick['현재가']:,.0f}원")
        c2.metric("점수", f"{int(pick['점수']):+d}")
        c3.metric("RSI", f"{pick['RSI']:.1f}")
        c4.metric("거래량 확인", "✅" if pick["거래량확인"] else "—")
        if st.button("🔍 이 종목 자세히 보기", key="today_pick_detail"):
            st.session_state["jump"] = (pick["종목명"], pick["종목코드"])
            st.session_state["_pending_view"] = VIEW_SINGLE_STOCK
            st.rerun()
        render_order_link(pick["종목명"], str(pick["종목코드"]))
    st.caption(
        f"사이드바 설정(임계값 ±{settings.threshold}) 기준, 시가총액 상위 "
        f"{TODAY_PICK_UNIVERSE}개 중 점수가 가장 높은 매수 후보입니다. 매수 신호가 "
        f"{len(buys)}개 있었는데 그중 1위이며, 매매 전 재무상태·뉴스는 직접 확인하세요."
    )
    _render_rescan_button()


def _render_scan_table(scan_result: pd.DataFrame) -> None:
    """같은 스캔 결과 전체를 표로 보여준다(신호 색상, 행 클릭 시 단일 종목 분석으로 이동, 파일 저장)."""
    st.divider()
    st.subheader(f"📋 스캔 결과 (시가총액 상위 {TODAY_PICK_UNIVERSE}개)")
    n_buy = (scan_result["신호"] == "매수").sum()
    n_sell = (scan_result["신호"] == "매도").sum()
    m1, m2, m3 = st.columns(3)
    m1.metric("스캔한 종목 수", len(scan_result))
    m2.metric("매수 신호", f"{n_buy}개")
    m3.metric("매도 신호", f"{n_sell}개")
    only_actionable = st.checkbox("매수/매도 신호만 보기", value=True, key="only_actionable")

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
        st.session_state["_pending_view"] = VIEW_SINGLE_STOCK
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

    st.caption("점수는 이평선/RSI/MACD/볼린저밴드 4개 지표 합산값입니다.")


def _render_watchlist_swap(scan_result: pd.DataFrame, watchlist: dict[str, str], settings: Settings) -> None:
    """관심종목 3개의 지금 점수와 스캔 결과 중 관심종목이 아닌 종목의 점수를 비교해 교체를 제안한다."""
    st.divider()
    st.subheader("💡 관심종목 교체 제안")
    st.caption(
        "현재 관심종목 3개의 '지금 시점 점수'와, 스캔 결과 중 관심종목이 아닌 종목의 "
        "점수를 비교합니다. 이 점수는 백테스트 성과가 아니라 오늘 시점 신호의 강도일 "
        "뿐이니 참고 정도로만 활용하세요."
    )

    current_scores = {}
    for name, code in watchlist.items():
        match = scan_result[scan_result["종목코드"] == code]
        if not match.empty:
            current_scores[name] = int(match.iloc[0]["점수"])
        else:
            fallback = prepare(code, 1, settings)
            current_scores[name] = int(fallback.iloc[-1]["SCORE_TOTAL"]) if fallback is not None else None

    candidates = scan_result[~scan_result["종목코드"].isin(watchlist.values())]
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
                new_watchlist = {k: v for k, v in watchlist.items() if k != weakest_name}
                new_watchlist[best_candidate["종목명"]] = best_candidate["종목코드"]
                save_watchlist(new_watchlist)
                st.success(
                    f"관심종목을 교체했습니다: {weakest_name} → {best_candidate['종목명']}. "
                    "'단일 종목 분석' 탭에서 바로 확인해보세요."
                )
                st.rerun()
        else:
            st.caption("현재 관심종목이 스캔 후보들보다 점수가 더 높거나 같아서 교체를 제안하지 않습니다.")
    elif candidates.empty:
        st.caption("스캔 대상 중 관심종목을 제외한 후보가 없습니다.")


def render(watchlist: dict[str, str], years: int, settings: Settings) -> None:
    st.caption(
        f"관심 종목 3개에 국한하지 않고, 시가총액 상위 {TODAY_PICK_UNIVERSE}개(KOSPI+KOSDAQ)를 자동으로 훑어서 "
        "**지금 시점에 매수/매도 신호가 뜬 종목**을 찾아줍니다. 탭을 열면 바로 스캔하고 추천 1개를 골라줍니다."
    )
    scan_result = _run_today_scan(settings)
    if scan_result is None:
        return
    _render_today_pick(scan_result, settings)
    if scan_result.empty:
        return
    _render_scan_table(scan_result)
    _render_watchlist_swap(scan_result, watchlist, settings)
