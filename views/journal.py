"""뷰: 매매 기록·수익률 — 직접 입력한 매수/매도 기록으로 일일 수익률을 자동 계산한다.

이 앱은 증권사(키움)와 연동되어 있지 않다. 체결 내역을 자동으로 가져오지 못하므로, 사용자가 체결 후
여기에 한 건씩 입력하면 평균단가법 실현손익과 종가 기준 일일 수익률(자금 유출입 반영)을 계산해 보여준다.
계산 방식은 strategy/journal.py 설명 참고.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.fetch import (
    fetch_ohlcv_range, load_holdings, load_trades, resolve_stock_name, save_holdings, save_trades,
)
from strategy.journal import (
    BUY, SELL, JournalError, apply_trade_to_holdings, daily_performance, opening_buy_for_sell, realized_pnl,
    summarize,
)


@st.cache_data(ttl=3600, show_spinner=False)
def _closes(code: str, start: str) -> pd.Series:
    df = fetch_ohlcv_range(code, start, dt.date.today().isoformat(), warmup_days=10)
    return df["Close"]


def _won(v: float) -> str:
    return f"{v:+,.0f}원"


def _render_entry_form() -> None:
    st.subheader("✍️ 매매 기록 입력")
    with st.form("trade_form", clear_on_submit=True):
        c1, c2, c3 = st.columns([1, 1, 1])
        date = c1.date_input("체결일", value=dt.date.today())
        code = c2.text_input("종목코드(6자리)", placeholder="005930")
        side = c3.radio("구분", [BUY, SELL], horizontal=True)
        c4, c5, c6 = st.columns(3)
        qty = c4.number_input("수량(주)", min_value=0.0, step=1.0, format="%g")
        price = c5.number_input("체결단가(원)", min_value=0.0, step=100.0, format="%g")
        fee = c6.number_input("수수료·세금 합계(원, 모르면 0)", min_value=0.0, step=10.0, format="%g")
        sync = st.checkbox("보유종목 표에도 자동 반영(매수는 추가·평균단가 갱신, 매도는 수량 차감/삭제)", value=True)
        submitted = st.form_submit_button("기록 추가", type="primary")

    if not submitted:
        return
    trade = {"날짜": date.isoformat(), "종목코드": code.strip().zfill(6), "구분": side,
             "수량": qty, "단가": price, "수수료세금": fee}
    try:
        trades = load_trades()
        holdings = load_holdings()
        added = []
        if side == SELL:  # 이 메뉴를 쓰기 전에 산 종목을 처음 팔면, 매입단가로 기초 매수를 함께 남긴다
            opening = opening_buy_for_sell(trades, holdings, trade["종목코드"], qty, trade["날짜"])
            if opening:
                added.append(opening)
        added.append(trade)
        # 새 기록을 넣었을 때 전체가 앞뒤가 맞는지(보유보다 많이 매도 등) 먼저 검증한다
        realized_pnl(trades + added)
        if sync:
            save_holdings(apply_trade_to_holdings(holdings, trade))
        save_trades(trades + added)
    except JournalError as e:
        st.error(str(e))
        return
    st.success("기록했습니다." + (" 보유종목에도 반영했습니다." if sync else ""))
    st.rerun()


def _render_trade_list() -> None:
    trades = load_trades()
    st.subheader("🧾 기록 목록")
    if not trades:
        st.caption("아직 기록이 없습니다. 위에서 체결 내역을 입력해 주세요.")
        return
    df = pd.DataFrame(trades)
    for col, default in [("수수료세금", 0.0)]:
        if col not in df.columns:
            df[col] = default
    df = df[["날짜", "종목코드", "구분", "수량", "단가", "수수료세금"]]
    edited = st.data_editor(
        df, num_rows="dynamic", use_container_width=True, key="trades_editor",
        column_config={
            "구분": st.column_config.SelectboxColumn("구분", options=[BUY, SELL], required=True),
            "수량": st.column_config.NumberColumn("수량(주)", min_value=0),
            "단가": st.column_config.NumberColumn("단가(원)", min_value=0),
            "수수료세금": st.column_config.NumberColumn("수수료·세금(원)", min_value=0),
        },
    )
    st.caption("행을 고치거나 지운 뒤 저장하세요. ⚠️ 여기서 고친 기록은 위 '보유종목 표'에 자동 반영되지 않습니다(보유종목은 따로 수정).")
    if st.button("💾 기록 저장"):
        clean = edited.dropna(subset=["날짜", "종목코드", "구분", "수량", "단가"])
        new_trades = [
            {"날짜": str(r["날짜"])[:10], "종목코드": str(r["종목코드"]).zfill(6), "구분": r["구분"],
             "수량": float(r["수량"]), "단가": float(r["단가"]), "수수료세금": float(r["수수료세금"] or 0)}
            for _, r in clean.iterrows()
        ]
        try:
            realized_pnl(new_trades)  # 검증
        except JournalError as e:
            st.error(str(e))
            return
        save_trades(new_trades)
        st.success("저장했습니다.")
        st.rerun()


def _render_results() -> None:
    trades = load_trades()
    st.subheader("📈 일일 수익률")
    if not trades:
        return
    try:
        first = min(str(t["날짜"])[:10] for t in trades)
        codes = sorted({str(t["종목코드"]).zfill(6) for t in trades})
        with st.spinner("종가를 불러와 수익률을 계산하는 중..."):
            closes = {c: _closes(c, first) for c in codes}
            perf = daily_performance(trades, closes)
            realized = realized_pnl(trades)
    except JournalError as e:
        st.error(str(e))
        return
    except Exception as e:  # noqa: BLE001 — 시세 조회 실패를 그대로 알린다
        st.error(f"시세를 불러오지 못했습니다: {type(e).__name__}: {e}")
        return

    if perf.empty:
        st.info("계산할 시세가 아직 없습니다.")
        return
    s = summarize(trades, perf)
    last = perf.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("오늘(최근 거래일) 수익률", f"{last['일수익률']:+.2%}", _won(last["일손익(원)"]))
    c2.metric("누적 수익률(시간가중)", f"{s['cum_return']:+.2%}")
    c3.metric("총 손익", _won(s["total_pnl"]), help="실현손익 + 평가손익(수수료·세금 반영)")
    c4.metric("실현손익 / 평가손익", f"{s['realized']:+,.0f} / {s['unrealized']:+,.0f}원")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=perf.index, y=perf["일수익률"] * 100, name="일일 수익률(%)", opacity=0.6))
    fig.add_trace(go.Scatter(x=perf.index, y=perf["누적수익률"] * 100, name="누적 수익률(%)", yaxis="y2", mode="lines"))
    fig.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="일일(%)",
        yaxis2=dict(title="누적(%)", overlaying="y", side="right"), legend=dict(orientation="h", y=-0.25),
    )
    st.plotly_chart(fig, use_container_width=True)

    show = perf.tail(30).iloc[::-1].copy()
    show.index = show.index.date
    show["일수익률"] = show["일수익률"].map(lambda v: f"{v:+.2%}")
    show["누적수익률"] = show["누적수익률"].map(lambda v: f"{v:+.2%}")
    for c in ["평가금액", "매수금액", "매도대금", "일손익(원)"]:
        show[c] = show[c].map(lambda v: f"{v:,.0f}")
    st.dataframe(show, use_container_width=True)
    if not realized.empty:
        with st.expander("매도별 실현손익 (평균단가법)"):
            r = realized.copy()
            r["날짜"] = r["날짜"].dt.date
            r["종목명"] = r["종목코드"].map(lambda c: resolve_stock_name(c) or c)
            r["수익률"] = r["수익률"].map(lambda v: f"{v:+.2%}")
            for c in ["매도가", "평균단가", "실현손익(원)"]:
                r[c] = r[c].map(lambda v: f"{v:,.0f}")
            st.dataframe(r[["날짜", "종목명", "종목코드", "수량", "매도가", "평균단가", "실현손익(원)", "수익률"]],
                         use_container_width=True, hide_index=True)
    st.caption(
        "계산 기준: 종가 평가, 자금 유출입(매수·매도)을 반영한 시간가중 근사입니다. 실제 증권사 수익률과 "
        "수수료·세금·배당·체결 시각 차이로 조금 다를 수 있고, 입력한 기록만 반영됩니다."
    )


def render() -> None:
    st.caption(
        "체결한 매수/매도를 한 건씩 입력하면 실현손익과 일일 수익률을 자동으로 계산합니다. "
        "이 앱은 키움증권과 연동되어 있지 않아, 체결 내역을 자동으로 가져오지는 못합니다."
    )
    _render_entry_form()
    st.divider()
    _render_results()
    st.divider()
    _render_trade_list()
