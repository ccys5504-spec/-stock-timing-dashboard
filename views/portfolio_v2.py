"""뷰 5: 포트폴리오 전략(v2) — "무엇을 살까"를 월 1회 정하는 실험적 전략.

이 화면은 두 가지를 보여준다:
  1) 지금 시점 기준 목표 포트폴리오(월말 종가 기준으로 확정해 다음 거래일 시가에 사는 방식의 미리보기)
  2) 이 전략을 상장폐지 종목까지 포함한 종목군에서 10년 백테스트한 결과(assets/v2_backtest.json)

⚠️ 2)의 결과가 중요하다: 생존편향을 줄인 종목군에서 이 전략은 코스피 단순보유에 크게 못 미친다.
그래서 이 화면은 추천 화면이 아니라 "실험 중인 전략과 그 검증 결과"를 함께 보여주는 화면이다.
주문은 이 앱이 대신 넣지 않는다.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import run_v2_portfolio
from data.fetch import load_holdings
from views.order_link import render_order_link

RESULT_PATH = Path(__file__).resolve().parent.parent / "assets" / "v2_backtest.json"


@st.cache_data(show_spinner=False)
def _load_backtest() -> dict | None:
    try:
        with open(RESULT_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1%}"


def _render_backtest(result: dict | None) -> None:
    st.subheader("📈 검증 결과 — 생존편향을 줄인 10년 백테스트")
    if result is None:
        st.info("백테스트 결과 파일(assets/v2_backtest.json)을 찾지 못했습니다.")
        return
    stats, curves = result["stats"], result["curves"]
    main_key = next((k for k in stats if k.startswith("A ")), None)
    if main_key is None:
        return
    main = stats[main_key]
    kospi = curves.get("코스피")
    ew = curves.get("동일가중(시점기준)")

    def total(series):
        return series[-1][1] / series[0][1] - 1 if series else None

    st.markdown(
        f"기간 2016-01 ~ {result['generated']} · 후보 = **그 시점** 시가총액 상위 {result['top_n']}개"
        f"(**상장폐지된 종목 포함**) · 그중 모멘텀 상위 {result['top_k']}개 보유 · 매월 말 리밸런싱 · 왕복비용 0.3%"
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("이 전략 (v2)", _fmt_pct(main["total"]), help=f"연복리 {_fmt_pct(main['cagr'])}, 최대낙폭 {_fmt_pct(main['mdd'])}")
    c2.metric("같은 종목군 동일가중 보유", _fmt_pct(total(ew)))
    c3.metric("코스피 단순보유", _fmt_pct(total(kospi)))
    st.caption(
        f"v2 연복리 {_fmt_pct(main['cagr'])} · 최대낙폭 {_fmt_pct(main['mdd'])} · 샤프 "
        f"{main['sharpe']:.2f}" if main.get("sharpe") is not None else ""
    )

    fig = go.Figure()
    names = {main_key: "v2 전략", "동일가중(시점기준)": "같은 종목군 동일가중 보유", "코스피": "코스피"}
    for key, label in names.items():
        series = curves.get(key)
        if series:
            fig.add_trace(go.Scatter(x=[p[0] for p in series], y=[p[1] for p in series], name=label, mode="lines"))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="누적 배수 (1.0 = 원금)",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "**어떻게 읽어야 하나요**\n"
        "- 이 전략은 같은 기간 코스피를 그냥 들고 있는 것보다 **크게 못 벌었습니다.** 종목군 전체를 동일가중으로 "
        "들고 있는 것과 비교해도 우위가 뚜렷하지 않습니다 — 즉 지금의 종목 선택(모멘텀 순위)이 실제로 "
        "값어치를 더한다는 증거가 약합니다.\n"
        "- 예전(생존편향 있는) 백테스트에서 +600%대가 나왔던 건, '오늘의 시총 상위 100개'라는 **이미 성공한 "
        "종목만 골라 담았기 때문**이었습니다. 과거 시점의 종목군에서 상장폐지 종목까지 포함해 다시 재니 결과가 "
        "크게 달라졌습니다.\n"
        "- 그래서 이 탭은 \"이걸 사세요\"가 아니라 **실험 중인 전략과 그 검증 결과**를 함께 보여주는 용도입니다."
    )
    with st.expander("전체 수치 / 시험 조건 자세히"):
        rows = []
        for key, st_ in stats.items():
            rows.append({
                "구성": key, "누적수익률": _fmt_pct(st_["total"]), "연복리": _fmt_pct(st_["cagr"]),
                "최대낙폭": _fmt_pct(st_["mdd"]),
                "샤프": f"{st_['sharpe']:.2f}" if st_.get("sharpe") is not None else "—",
                "매매(회)": st_["trades"],
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.markdown(
            "- **A**: 현재 기본 설정(장세필터 켬, ATR 손절 끔). A2/A3은 상장폐지 종목 회수액을 절반으로 가정 / 비용을 "
            "1.0%로 올린 민감도. **B**: 지금도 상장 중인 종목만(폐지 제외). **C**: 이전 설정(장세필터 끔·ATR 손절).\n"
            "- 시가총액은 '보정된 종가 × 상장주식수'로 근사했고 과거 증자/소각 이력은 반영하지 못해, 일부 종목의 과거 "
            "순위가 실제와 다를 수 있습니다. 시장충격·상하한가 체결불가는 반영되지 않았습니다.\n"
            "- 재현: `python scripts/build_pit_dataset.py` 후 `python scripts/portfolio_pit_backtest.py`."
        )


def _render_target(held_codes: tuple[str, ...]) -> None:
    st.subheader("🎯 지금 기준 목표 포트폴리오 (미리보기)")
    with st.spinner("시가총액 상위 100개의 시세를 불러와 순위를 계산하는 중... (처음엔 1분쯤 걸릴 수 있습니다)"):
        try:
            res = run_v2_portfolio(held_codes)
        except Exception as e:  # noqa: BLE001 — 외부 시세 조회 실패를 화면에 그대로 알린다
            st.error(f"목표 포트폴리오를 계산하지 못했습니다: {type(e).__name__}: {e}")
            return
    table, as_of = res["table"], res["as_of"]
    st.caption(
        f"조회 성공 {res['loaded']}/{res['requested']}개"
        + (f" · 데이터 기준일 {as_of.date().isoformat()}" if as_of is not None else "")
        + " · 이 표는 최신 종가 기준 미리보기이고, 전략의 기준은 **매월 마지막 거래일 종가**입니다."
    )
    if res["failed"]:
        st.warning(f"시세 조회에 실패한 종목 {len(res['failed'])}개는 계산에서 빠졌습니다: "
                   + ", ".join(n for _, n in res["failed"][:10]))
    if res["index_missing"]:
        st.warning("지수 시세를 불러오지 못해 장세 배율을 100%로 가정했습니다: " + ", ".join(res["index_missing"]))

    cols = st.columns(len(res["regime"]) or 1)
    for col, (market, (mult, label)) in zip(cols, res["regime"].items()):
        col.metric(f"{market} 장세", f"투자비중 {mult:.0%}", help=label)
        col.caption(label)

    if table.empty:
        st.info("지금 조건을 만족하는 후보가 없습니다.")
        return
    invested = table["목표비중(%)"].sum()
    st.markdown(f"목표 투자 비중 합계 **{invested:.0f}%** (나머지 {100 - invested:.0f}%는 현금)")
    show = table.copy()
    show["현재가"] = show["현재가"].map(lambda v: f"{v:,.0f}")
    show["손절참고가"] = show["손절참고가"].map(lambda v: "—" if pd.isna(v) else f"{v:,.0f}")
    st.dataframe(show, hide_index=True, use_container_width=True)
    if res["dropped"]:
        st.info("보유 중이지만 이번 목표에서 빠진 종목(월말 정리 후보): " + ", ".join(n for _, n in res["dropped"]))
    st.caption(
        "손절참고가 = 현재가 − 2.5×ATR(변동성). 검증에서 이 손절을 전략에 넣으면 수익이 크게 줄어 **전략 규칙에는 "
        "넣지 않았고**, 방어용으로 직접 쓰실 때의 참고값으로만 표시합니다."
    )
    with st.expander("이 종목을 사고 싶다면"):
        pick = st.selectbox("종목", table["종목명"] + " (" + table["종목코드"] + ")", key="v2_pick")
        code = pick.rsplit("(", 1)[1].rstrip(")")
        render_order_link(pick.rsplit(" (", 1)[0], code)


def render() -> None:
    st.caption(
        "매월 말에 시가총액 상위 100개 중 최근 3·6·12개월 상승률이 강한 15종목을 변동성이 낮을수록 많이 담고, "
        "시장이 약하면(지수가 200일선 아래) 투자 비중을 줄이는 실험적 전략입니다. 사이드바 설정은 이 탭에 적용되지 않습니다."
    )
    _render_backtest(_load_backtest())
    st.divider()
    holdings = load_holdings()
    held = tuple(sorted({str(h["종목코드"]) for h in holdings if h.get("종목코드")}))
    _render_target(held)
    with st.expander("규칙 자세히"):
        st.markdown(
            "1. **후보**: 그 시점 시가총액 상위 100개(우선주·스팩·거래대금 5억 미만 제외)\n"
            "2. **순위**: 최근 3/6/12개월 수익률 평균(최근 1개월은 제외 — 단기 반전 효과 회피)\n"
            "3. **선택**: 상위 15개. 이미 보유 중인 종목은 순위가 30위 밖으로 밀릴 때만 교체\n"
            "4. **비중**: 변동성 역가중, 종목당 12% 상한\n"
            "5. **장세**: 지수가 200일선 위·기울기 상승이면 100%, 위·기울기 하락 75%, 아래·상승 40%, 아래·하락 20% 투자\n"
            "6. **체결**: 월말 종가로 정하고 다음 거래일 시가에 매매(이 앱은 주문하지 않습니다)\n"
            "7. 비중 차이가 1.5% 미만인 소액 조정은 생략"
        )
