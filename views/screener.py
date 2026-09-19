"""뷰 3: 종목 추천 (스크리너)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from core import VIEW_SINGLE_STOCK, Settings, prepare, run_screener, to_csv_bytes, to_excel_bytes
from data.fetch import save_watchlist
from views.order_link import render_order_link


TODAY_PICK_UNIVERSE = 100  # "오늘의 추천"이 훑어보는 시가총액 상위 종목 수 (고정)
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


def _render_today_pick(settings: Settings) -> None:
    """스캔 버튼을 누를 필요 없이, 시가총액 상위 100개 중 오늘 시점 점수가
    가장 높은 매수 후보 1개를 자동으로 보여준다 — 매일 한 번 열어보는 용도.
    run_screener()가 30분 캐싱돼 있어서 같은 설정으로 여러 번 열어봐도
    매번 새로 스캔하지 않는다.
    """
    st.markdown("### ⭐ 오늘의 추천")
    with st.spinner(f"시가총액 상위 {TODAY_PICK_UNIVERSE}개 종목을 훑어보는 중..."):
        try:
            today_result = run_screener(
                ["KOSPI", "KOSDAQ"], TODAY_PICK_UNIVERSE,
                settings.threshold, settings.volume_filter_mode, settings.volume_multiplier,
            )
        except Exception:  # noqa: BLE001
            st.warning("오늘의 추천을 계산하지 못했습니다(데이터 조회 실패). 아래에서 직접 스캔해보세요.")
            return

    _render_scan_health(today_result)
    if today_result.empty:
        st.caption("스캔 결과가 없습니다.")
        return

    buys = today_result[today_result["신호"] == "매수"]
    if buys.empty:
        st.info(
            f"오늘은 시가총액 상위 {TODAY_PICK_UNIVERSE}개 종목 중 매수 신호가 뜬 종목이 "
            "없습니다. 신호가 없으면 관망하는 것도 하나의 선택입니다."
        )
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


def render(watchlist: dict[str, str], years: int, settings: Settings) -> None:
    st.caption(
        "관심 종목 3개에 국한하지 않고, 시가총액 상위 종목들을 훑어서 "
        "**지금 시점에 매수/매도 신호가 뜬 종목**을 찾아줍니다."
    )

    _render_today_pick(settings)
    st.divider()
    st.subheader("직접 범위를 정해서 스캔하기")

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
                    scan_markets, scan_top_n, settings.threshold,
                    settings.volume_filter_mode, settings.volume_multiplier,
                )
            st.session_state["scan_result"] = scan_result

    scan_result = st.session_state.get("scan_result")
    if scan_result is None:
        st.caption("👆 '스캔 시작'을 누르면 결과가 여기에 표시됩니다.")
    elif scan_result.empty:
        st.warning("스캔 결과가 없습니다. 데이터 조회에 실패했을 수 있습니다.")
        _render_scan_health(scan_result)
    else:
        n_buy = (scan_result["신호"] == "매수").sum()
        n_sell = (scan_result["신호"] == "매도").sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("스캔한 종목 수", len(scan_result))
        m2.metric("매수 신호", f"{n_buy}개")
        m3.metric("매도 신호", f"{n_sell}개")
        _render_scan_health(scan_result)

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
            st.caption("스캔 대상 중 관심종목을 제외한 후보가 없습니다 (스캔 범위를 넓혀보세요).")
