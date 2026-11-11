"""뷰 4: 내 보유종목.

2026-09-18: "3종목 비교" 탭을 없애고, 그 기능(여러 종목의 백테스트 성과를
나란히 비교)을 이 탭 안으로 옮겼다 — 대신 관심종목 3개 고정이 아니라
**실제 보유 중인 종목들**을 자동으로 불러와서 비교한다.
"""
from __future__ import annotations

import datetime as dt
import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.engine import buy_and_hold_return_pct, run_backtest
from core import Settings, load_data, prepare
from data.fetch import load_holdings, resolve_stock_name, save_holdings
from strategy.exit_rules import DEFAULT_RULE, evaluate_holding
from views.order_link import render_order_link


def render(years: int, settings: Settings) -> None:
    st.caption(
        "실제로 보유 중인 종목의 수량·매입단가를 입력해두면, 현재가 대비 평가손익과 "
        "지금 시점 매수/매도 신호, 그리고 전략 백테스트 비교까지 한 화면에서 확인할 수 있습니다."
    )
    st.info(
        "ℹ️ 여기 입력한 정보는 비공개 저장소에만 저장됩니다(이 앱을 초대받은 "
        "사람만 접근 가능한 GitHub 비공개 Gist, 또는 그게 설정 안 된 환경에서는 "
        "이 컴퓨터의 data/holdings.json 파일). 광고·분석 등 다른 용도로 쓰이지 "
        "않습니다."
    )

    holdings = load_holdings()
    editor_cols = ["종목코드", "수량", "매입단가", "매수일"]
    holdings_df = pd.DataFrame(holdings) if holdings else pd.DataFrame(columns=editor_cols)
    for col in editor_cols:
        if col not in holdings_df.columns:
            holdings_df[col] = pd.Series(dtype="object")

    st.subheader("보유종목 입력")
    edited_df = st.data_editor(
        holdings_df[editor_cols],
        num_rows="dynamic",
        use_container_width=True,
        key="holdings_editor",
        column_config={
            "종목코드": st.column_config.TextColumn("종목코드", help="6자리 종목코드, 예: 000660", width="small"),
            "수량": st.column_config.NumberColumn("수량(주)", min_value=0, step=1),
            "매입단가": st.column_config.NumberColumn("매입단가(원)", min_value=0, step=100),
            "매수일": st.column_config.TextColumn(
                "매수일(선택)", width="small",
                help="YYYY-MM-DD. 입력하면 '매도 규칙 점검'이 매수 이후의 고점을 정확히 계산합니다(비우면 오늘 봉만 봅니다)."),
        },
    )

    if st.button("💾 저장", type="primary"):
        clean = edited_df.dropna(subset=["종목코드"])
        clean = clean[clean["종목코드"].astype(str).str.strip() != ""]
        save_holdings(_clean_records(clean))
        st.success("저장했습니다.")
        st.rerun()

    saved_holdings = load_holdings()
    if not saved_holdings:
        st.caption("아직 입력된 보유종목이 없습니다. 위 표에 입력 후 저장을 눌러주세요.")
        st.stop()

    _render_sell_complete(saved_holdings)

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
            df = prepare(code, 1, settings)
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
            if settings.stop_loss_pct is not None and pnl_pct <= -settings.stop_loss_pct:
                stoploss_alerts.append(f"{name} ({pnl_pct:+.1%})")

    rule_rows = _exit_rule_rows(saved_holdings)
    rule_alerts = [r for r in rule_rows if r.get("발동")]
    if rule_alerts:
        st.error(
            "🚨 **매도 규칙에 닿은 종목**: " + ", ".join(
                f"{r['종목명']}({r['규칙']} 도달, 일봉 기준 {r['발동일']})" for r in rule_alerts)
            + " — 매도 여부는 직접 판단·주문하세요(이 앱은 주문하지 않습니다)."
        )
    if sell_alerts:
        st.warning(f"🔔 보유 중인 종목 중 **매도 신호**가 뜬 종목: {', '.join(sell_alerts)}")
    if stoploss_alerts:
        st.error(
            f"⚠️ 사이드바 손절 기준(-{settings.stop_loss_pct:.0%})에 도달한 종목: {', '.join(stoploss_alerts)}"
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
    _render_exit_rule_table(rule_rows)
    render_order_link()  # 매도/추가매수를 하려면 여기서 키움 WTS로 이동

    st.divider()
    _render_strategy_comparison(saved_holdings, years, settings)


def _clean_records(df: pd.DataFrame) -> list[dict]:
    """편집표를 저장용 기록으로 바꾼다. 비어 있는 매수일(NaN)은 키를 아예 빼서 JSON에 NaN이 들어가지 않게 한다."""
    records = []
    for rec in df.to_dict("records"):
        out = {k: v for k, v in rec.items() if k != "매수일"}
        date = rec.get("매수일")
        if isinstance(date, str) and date.strip():
            try:
                out["매수일"] = pd.Timestamp(date.strip()).date().isoformat()
            except (ValueError, TypeError):
                pass  # 날짜로 읽을 수 없으면 저장하지 않는다(나중에 다시 입력)
        records.append(out)
    return records


def _exit_rule_rows(saved_holdings: list[dict]) -> list[dict]:
    """보유종목마다 사용자 매도 규칙(손절 -3%, +5/+10/+20% 이익 보호선)을 일봉으로 점검한다."""
    rows = []
    for h in saved_holdings:
        code = str(h.get("종목코드", "")).strip().zfill(6)
        buy_price = float(h.get("매입단가") or 0)
        if not code or buy_price <= 0:
            continue
        name = resolve_stock_name(code) or code
        entry = h.get("매수일") or None
        try:
            years = 1
            if entry:
                years = min(10, max(1, math.ceil((dt.date.today() - pd.Timestamp(entry).date()).days / 365) + 1))
            res = evaluate_holding(load_data(code, years), buy_price, entry, DEFAULT_RULE)
        except Exception as e:  # noqa: BLE001 — 한 종목 실패가 화면 전체를 막지 않게
            rows.append({"종목명": name, "오류": f"{type(e).__name__}: {e}"})
            continue
        rows.append({
            "종목명": name, "매입가": buy_price, "현재가": res["last_price"], "수익률": res["last_return"],
            "고점수익률": res["peak_return"], "매도선가": res["floor_price"], "매도선": res["floor_return"],
            "발동": res["triggered"], "규칙": res["label"],
            "발동일": res["exit_date"].date().isoformat() if res["exit_date"] is not None else "",
            "정확": res["precise"],
        })
    return rows


def _render_exit_rule_table(rule_rows: list[dict]) -> None:
    st.subheader("📏 매도 규칙 점검")
    st.caption(
        "내가 정한 규칙: **매입가 대비 -3%에 닿으면 매도**, 고점 수익률이 **+5% / +10% / +20%**를 넘으면 그 구간의 "
        "가격이 새 매도선(예: +12%까지 올랐다가 +10%로 내려오면 매도). 한 번 올라간 매도선은 내려가지 않습니다. "
        "일봉으로 확인하는 값이라 장중 실시간이 아니고, **알림일 뿐 자동으로 주문하지 않습니다.**"
    )
    shown = [r for r in rule_rows if "오류" not in r]
    for r in rule_rows:
        if "오류" in r:
            st.warning(f"{r['종목명']}: 규칙을 점검하지 못했습니다({r['오류']}).")
    if not shown:
        return
    table = pd.DataFrame([{
        "종목명": r["종목명"], "매입가": f"{r['매입가']:,.0f}", "현재가": f"{r['현재가']:,.0f}",
        "수익률": f"{r['수익률']:+.1%}", "고점 수익률": f"{r['고점수익률']:+.1%}",
        "현재 매도선": f"{r['매도선가']:,.0f} ({r['매도선']:+.0%})",
        "상태": (f"🚨 {r['규칙']} 도달 ({r['발동일']})" if r["발동"] else f"유지 — {r['규칙']}"),
        "계산 범위": "매수일 이후" if r["정확"] else "오늘 봉만(매수일 미입력)",
    } for r in shown])
    st.dataframe(table, use_container_width=True, hide_index=True)
    if any(not r["정확"] for r in shown):
        st.caption("💡 매수일을 입력하면 매수 이후의 고점을 정확히 반영합니다. 비워두면 이익 구간 매도선이 실제보다 낮게 나올 수 있습니다.")


def _current_price(code: str) -> float | None:
    """이 종목의 가장 최근 종가. core.load_data가 30분 보관하므로 아래 '현재 상태' 표와 같은 값을 쓴다."""
    try:
        df = load_data(code, 1)
    except Exception:  # noqa: BLE001 — 조회 실패 시 참고가만 없이 넘어간다(삭제 자체는 계속 가능해야 함)
        return None
    return float(df["Close"].iloc[-1]) if df is not None and not df.empty else None


def _render_sell_complete(saved_holdings: list[dict]) -> None:
    """종목마다 '매도 완료(삭제)' 버튼 — 누르면(한 번 더 확인 후) 보유종목에서 지우고 저장한다.

    이 앱은 증권사와 연동되지 않아서, 실제 매도는 키움에서 직접 한 뒤 여기서 정리하는 용도다.
    (2026-10-16: 예전에는 매도가를 입력하면 '매매 기록·수익률' 탭에 기록을 남겼는데, 그 탭을 없애서
    매도가 입력과 기록 저장도 함께 없앴다. 각 행의 현재가·평가손익률 표시는 그대로 둔다.)
    """
    st.subheader("✅ 매도 완료 처리")
    st.caption(
        "키움에서 실제로 매도한 종목은 여기서 정리하세요. 버튼을 누르면 한 번 더 확인한 뒤 보유종목 표에서 "
        "삭제합니다(매매 내역은 따로 저장하지 않습니다)."
    )
    for h in saved_holdings:
        code = str(h.get("종목코드", "")).strip().zfill(6)
        qty = float(h.get("수량") or 0)
        cost = float(h.get("매입단가") or 0)
        if not code or qty <= 0:
            continue
        name = resolve_stock_name(code) or code
        cur_price = _current_price(code)
        confirm_key = f"confirm_sell_{code}"

        if st.session_state.get(confirm_key):
            st.warning(f"**{name}** ({code}) · {qty:g}주를 정말 삭제할까요? 되돌릴 수 없습니다.")
            cc1, cc2 = st.columns(2)
            if cc1.button("확인, 삭제합니다", key=f"sellconfirm_{code}", type="primary",
                          use_container_width=True):
                remaining = [x for x in load_holdings()
                             if str(x.get("종목코드", "")).strip().zfill(6) != code]
                save_holdings(remaining)
                st.session_state.pop(confirm_key, None)
                st.success(f"{name}을(를) 보유종목에서 삭제했습니다.")
                st.rerun()
            if cc2.button("취소", key=f"sellcancel_{code}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.rerun()
            continue

        c1, c3 = st.columns([3.8, 1.4])
        price_line = f"매입 {cost:,.0f}원"
        if cur_price is not None:
            pnl_pct = cur_price / cost - 1 if cost > 0 else None
            price_line += f" · 현재가 {cur_price:,.0f}원" + (f" ({pnl_pct:+.1%})" if pnl_pct is not None else "")
        else:
            price_line += " · 현재가 조회 실패"
        c1.markdown(f"**{name}** ({code}) · {qty:g}주 · {price_line}")
        if c3.button("매도 완료(삭제)", key=f"sellbtn_{code}", use_container_width=True):
            st.session_state[confirm_key] = True
            st.rerun()


def _render_strategy_comparison(saved_holdings: list[dict], years: int, settings: Settings) -> None:
    """보유종목들의 백테스트 성과를 나란히 비교한다.

    예전 "3종목 비교" 탭이 하던 일과 같지만, 관심종목 3개 고정이 아니라
    지금 입력해둔 보유종목을 자동으로 불러온다(중복 종목코드는 한 번만).
    """
    st.subheader("📊 보유종목 전략 백테스트 비교")

    name_to_code: dict[str, str] = {}
    seen_codes: set[str] = set()
    for h in saved_holdings:
        code = str(h.get("종목코드", "")).strip().zfill(6)
        if not code or code in seen_codes:
            continue
        seen_codes.add(code)
        name_to_code[resolve_stock_name(code) or code] = code

    if not name_to_code:
        st.caption("비교할 보유종목이 없습니다. 위에 종목코드를 입력하고 저장해보세요.")
        return

    st.caption("사이드바에서 설정한 조회 기간·임계값·거래량 필터·손절 규칙이 아래 종목에 동일하게 적용됩니다.")

    rows = []
    equity_curves = {}
    signal_badges = {}

    with st.spinner("보유종목별 데이터를 불러오고 백테스트하는 중..."):
        for name, code in name_to_code.items():
            df = prepare(code, years, settings)
            if df is None:
                rows.append({"종목": name, "오류": "데이터 없음"})
                continue

            result = run_backtest(
                df, stop_loss_pct=settings.stop_loss_pct, trailing_stop_pct=settings.trailing_stop_pct,
                exit_on_signal=settings.exit_on_signal, adaptive_exit=settings.adaptive_exit,
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

    cols = st.columns(len(name_to_code))
    for col, name in zip(cols, name_to_code):
        sig = signal_badges.get(name, "—")
        color = {"매수": "green", "매도": "red", "관망": "gray"}.get(sig, "gray")
        with col:
            st.markdown(f"**{name}**")
            st.markdown(f"### :{color}[● {sig}]")

    st.markdown("**백테스트 요약 비교**")
    display_df = summary_df.copy()
    for pct_col in ["전략 수익률", "단순보유 수익률", "MDD", "승률"]:
        if pct_col in display_df.columns:
            display_df[pct_col] = display_df[pct_col].apply(
                lambda v: f"{v:+.1%}" if pd.notna(v) else "—"
            )
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if "전략 수익률" in summary_df and summary_df["전략 수익률"].notna().any():
        best = summary_df.loc[summary_df["전략 수익률"].idxmax(), "종목"]
        st.caption(f"💡 이 설정·기간 기준으로는 **{best}**의 전략 수익률이 가장 높았습니다.")

    if equity_curves:
        st.markdown("**전략 누적수익률 곡선 비교 (시작점 = 1.0)**")
        fig = go.Figure()
        for name, curve in equity_curves.items():
            fig.add_trace(go.Scatter(x=curve.index, y=curve.values, name=name, mode="lines"))
        fig.update_layout(height=450, yaxis_title="누적 배수 (1.0 = 원금)",
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "⚠️ 여러 조합을 시험해보며 '가장 수익률 높은 설정'을 찾는 것 자체가 과최적화(overfitting) "
        "위험이 있습니다. 과거 데이터 한 번에 잘 맞는 설정이 미래에도 맞는다는 보장은 없습니다 — "
        "상승장/하락장 등 서로 다른 기간에서 두루 검증(scripts/optimize.py)해보고 극단적으로 "
        "낙관적인 조합은 의심하는 게 안전합니다."
    )
