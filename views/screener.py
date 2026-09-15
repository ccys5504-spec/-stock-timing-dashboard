"""뷰 3: 종목 추천 (스크리너)."""
from __future__ import annotations

import datetime as dt

import pandas as pd
import streamlit as st

from core import VIEWS, Settings, prepare, run_screener, to_csv_bytes, to_excel_bytes
from data.fetch import save_watchlist


def render(watchlist: dict[str, str], years: int, settings: Settings) -> None:
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
                    scan_markets, scan_top_n, settings.threshold,
                    settings.volume_filter_mode, settings.volume_multiplier,
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
                        "'3종목 비교' 탭에서 바로 확인해보세요."
                    )
                    st.rerun()
            else:
                st.caption("현재 관심종목이 스캔 후보들보다 점수가 더 높거나 같아서 교체를 제안하지 않습니다.")
        elif candidates.empty:
            st.caption("스캔 대상 중 관심종목을 제외한 후보가 없습니다 (스캔 범위를 넓혀보세요).")
