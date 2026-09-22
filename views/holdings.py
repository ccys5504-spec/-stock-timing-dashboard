"""뷰 4: 내 보유종목.

2026-09-18: "3종목 비교" 탭을 없애고, 그 기능(여러 종목의 백테스트 성과를
나란히 비교)을 이 탭 안으로 옮겼다 — 대신 관심종목 3개 고정이 아니라
**실제 보유 중인 종목들**을 자동으로 불러와서 비교한다.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.engine import buy_and_hold_return_pct, run_backtest
from core import Settings, load_data, prepare
from data.fetch import load_holdings, load_trades, resolve_stock_name, save_holdings, save_trades
from strategy.journal import SELL, JournalError, opening_buy_for_sell, realized_pnl
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
    render_order_link()  # 매도/추가매수를 하려면 여기서 키움 WTS로 이동

    st.divider()
    _render_strategy_comparison(saved_holdings, years, settings)


def _current_price(code: str) -> float | None:
    """이 종목의 가장 최근 종가. core.load_data가 1시간 캐싱하므로 아래 '현재 상태' 표와 같은 값을 쓴다."""
    try:
        df = load_data(code, 1)
    except Exception:  # noqa: BLE001 — 조회 실패 시 참고가만 없이 넘어간다(삭제 자체는 계속 가능해야 함)
        return None
    return float(df["Close"].iloc[-1]) if df is not None and not df.empty else None


def _render_sell_complete(saved_holdings: list[dict]) -> None:
    """종목마다 '매도 완료(삭제)' 버튼 — 누르면 보유종목에서 지우고 저장한다.

    매도가를 넣으면 매매 기록(📒 메뉴)에도 매도로 남겨서 실현손익·일일 수익률에 반영한다.
    이 앱은 증권사와 연동되지 않아서, 실제 매도는 키움에서 직접 한 뒤 여기서 정리하는 용도다.
    매도가 입력칸은 참고용으로 현재가를 기본값으로 채워두되, 실제 체결가와 다르면 직접 고칠 수 있다.
    """
    st.subheader("✅ 매도 완료 처리")
    st.caption(
        "키움에서 실제로 매도한 종목은 여기서 정리하세요. 매도가는 참고용으로 현재가가 기본 입력되어 있고, "
        "실제 체결가로 바꿀 수 있습니다. 매도가를 입력하면 매매 기록에도 남아 "
        "'📒 매매 기록·수익률' 메뉴의 실현손익·일일 수익률에 반영됩니다(0이면 삭제만 합니다)."
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
        price_key = f"confirm_sell_price_{code}"

        if st.session_state.get(confirm_key):
            price = float(st.session_state.get(price_key, 0.0) or 0.0)
            st.warning(f"**{name}** ({code}) · {qty:g}주를 정말 삭제할까요? 되돌릴 수 없습니다.")
            cc1, cc2 = st.columns(2)
            if cc1.button("확인, 삭제합니다", key=f"sellconfirm_{code}", type="primary",
                          use_container_width=True):
                if price > 0:
                    trades = load_trades()
                    today = dt.date.today().isoformat()
                    added = []
                    opening = opening_buy_for_sell(trades, load_holdings(), code, qty, today)
                    if opening:
                        added.append(opening)  # 기록에 매수 이력이 없으면 매입단가로 기초 매수를 함께 남긴다
                    added.append({"날짜": today, "종목코드": code, "구분": SELL, "수량": qty,
                                  "단가": price, "수수료세금": 0.0})
                    try:
                        realized_pnl(trades + added)
                    except JournalError as e:
                        st.error(str(e))
                        st.stop()
                    save_trades(trades + added)
                remaining = [x for x in load_holdings()
                             if str(x.get("종목코드", "")).strip().zfill(6) != code]
                save_holdings(remaining)
                st.session_state.pop(confirm_key, None)
                st.session_state.pop(price_key, None)
                st.success(f"{name}을(를) 보유종목에서 삭제했습니다." + (" 매도 기록도 남겼습니다." if price > 0 else ""))
                st.rerun()
            if cc2.button("취소", key=f"sellcancel_{code}", use_container_width=True):
                st.session_state.pop(confirm_key, None)
                st.session_state.pop(price_key, None)
                st.rerun()
            continue

        c1, c2, c3 = st.columns([2.2, 1.6, 1.4])
        price_line = f"매입 {cost:,.0f}원"
        if cur_price is not None:
            pnl_pct = cur_price / cost - 1 if cost > 0 else None
            price_line += f" · 현재가 {cur_price:,.0f}원" + (f" ({pnl_pct:+.1%})" if pnl_pct is not None else "")
        else:
            price_line += " · 현재가 조회 실패"
        c1.markdown(f"**{name}** ({code}) · {qty:g}주 · {price_line}")
        price = c2.number_input("매도가(원)", min_value=0.0, step=100.0, format="%g",
                                value=cur_price if cur_price is not None else 0.0,
                                key=f"sellpx_{code}", label_visibility="collapsed",
                                help="체결된 매도 단가. 기본값은 현재가이며, 실제 체결가로 바꾸세요. 0이면 기록 없이 삭제만 합니다.")
        if c3.button("매도 완료(삭제)", key=f"sellbtn_{code}", use_container_width=True):
            st.session_state[confirm_key] = True
            st.session_state[price_key] = price
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
