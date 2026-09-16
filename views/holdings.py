"""뷰 4: 내 보유종목."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core import Settings, prepare
from data.fetch import load_holdings, resolve_stock_name, save_holdings


def render(settings: Settings) -> None:
    st.caption(
        "실제로 보유 중인 종목의 수량·매입단가를 입력해두면, 현재가 대비 평가손익과 "
        "지금 시점 매수/매도 신호를 한 화면에서 확인할 수 있습니다."
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
