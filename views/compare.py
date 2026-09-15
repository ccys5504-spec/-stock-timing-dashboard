"""뷰 2: 3종목 비교."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtest.engine import buy_and_hold_return_pct, run_backtest
from core import Settings, prepare


def render(watchlist: dict[str, str], years: int, settings: Settings) -> None:
    st.caption("사이드바에서 설정한 임계값/거래량 필터/손절 규칙이 아래 종목에 동일하게 적용됩니다.")

    rows = []
    equity_curves = {}
    signal_badges = {}

    with st.spinner("종목별 데이터를 불러오고 백테스트하는 중..."):
        for name, code in watchlist.items():
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

    cols = st.columns(len(watchlist))
    for col, (name, _) in zip(cols, watchlist.items()):
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
