"""v2 전략의 "지금 시점 목표 포트폴리오" 계산 — 앱 화면(views/portfolio_v2.py)이 사용한다.

백테스트 엔진(backtest/portfolio_engine.py)이 매월 말에 하는 일을 지금 시점 한 번만 한다:
  1) 후보 = 지금 시가총액 상위 100개(백테스트의 '그 시점 시총 상위 100개' 규칙과 같다. 호출하는
     쪽이 이미 상위 100개만 넘기므로 여기서 더 좁히지 않는다)
  2) 후보 정렬: 기본은 시가총액 큰 순(3종목 검증에서 모멘텀보다 훨씬 좋았음), ranking="momentum"이면 3/6/12개월 상대강도
  3) 상위 top_k(기본 3)개. 이미 보유 중인 종목은 순위가 top_k의 2배 밖으로 밀려야 제외(순위 이력)
  4) 변동성 역가중 비중(종목당 상한 = max(12%, 1.5/top_k))
  5) 참고용 초기 손절가 = 현재가 - 2.5 x ATR

장세필터(지수 200일선 기준 총 투자비중 100/75/40/20%)는 백테스트 기본값과 같이 적용하고, ATR
손절은 백테스트 기본값에서 꺼져 있어(켜면 생존편향 없는 종목군에서 수익이 크게 줄었다) 손절선은
'참고용'으로만 보여준다. 주문은 내지 않는다 — 이 표는 "월말 종가 기준 목표"이고, 실제 매매는 사용자가 다음 거래일 시가 근처에서
직접 한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.portfolio_engine import (
    ATR_INITIAL_MULT, RANK_CUTOFF_MULTIPLE, TOP_K, inverse_vol_weights, regime_multiplier,
    select_with_hysteresis, weight_cap, DEFAULT_RANKING, reorder_candidates,
)
from signals.indicators import compute_atr
from strategy.momentum import rank_universe

STATUS_NOT_HELD = "목표에 포함 (미보유)"
STATUS_HELD = "목표에 포함 (보유 중)"

UNIVERSE_TOP_N = None  # 넘겨받은 종목이 곧 후보(시가총액 상위 100개)

REGIME_LABELS = {
    1.00: "상승 추세 (지수가 200일선 위, 기울기 상승)",
    0.75: "주의 (200일선 위지만 기울기 하락)",
    0.40: "약세 (200일선 아래, 기울기 상승)",
    0.20: "하락 추세 (200일선 아래, 기울기 하락)",
}


def current_regime(index_data: dict[str, pd.DataFrame]) -> dict[str, tuple[float, str]]:
    """시장별 (총 투자비중 배율, 설명). 지수 시세가 없으면 배율 1.0(중립)으로 둔다."""
    out = {}
    for market, df in index_data.items():
        mult = regime_multiplier(df) if df is not None and len(df) else 1.0
        out[market] = (mult, REGIME_LABELS.get(mult, ""))
    return out


def build_target_portfolio(
    price_data: dict[str, pd.DataFrame],
    names: dict[str, str],
    markets: dict[str, str],
    held_codes: set[str] | None = None,
    regime: dict[str, tuple[float, str]] | None = None,
    top_k: int = TOP_K,
    universe_top_n: int | None = UNIVERSE_TOP_N,
    marcap: dict[str, float] | None = None,
    ranking: str | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """(목표 포트폴리오 표, 기준일)을 돌려준다. 후보가 없으면 (빈 표, None).

    표 컬럼: 순위, 종목코드, 종목명, 시장, 모멘텀(%), 목표비중(%), 현재가, 손절참고가, 상태
    목표비중은 변동성 역가중 비중 x 그 종목 시장의 장세 배율이라, 합계가 100% 미만이면 나머지는 현금.
    상태: '목표에 포함 (미보유)' / '목표에 포함 (보유 중)' — 내 보유종목과 비교한 표시일 뿐 매수·매도 권유가 아니다.
    """
    held_codes = held_codes or set()
    latest_dates = [df.index[-1] for df in price_data.values() if len(df)]
    if not latest_dates:
        return pd.DataFrame(), None
    as_of = max(latest_dates)

    ranked = rank_universe(price_data, as_of, top_n=universe_top_n)
    if ranked.empty:
        return pd.DataFrame(), None

    if ranking is None:  # 시가총액 정보가 있으면 기본(시총 순), 없으면 모멘텀
        ranking = DEFAULT_RANKING if marcap else "momentum"
    if ranking != "momentum":
        # 시가총액 순 등 다른 기준: 시총은 (현재 시가총액 / 현재가) = 주식수 x 종가로 근사
        implied_shares = {
            c: (marcap[c] / float(price_data[c]["Close"].iloc[-1]))
            for c in ranked["코드"] if marcap and c in marcap
        }
        ranked = reorder_candidates(ranked, ranking, price_data, as_of, implied_shares or None, np.random.default_rng(0))
    target_codes = select_with_hysteresis(ranked, top_k, top_k * RANK_CUTOFF_MULTIPLE, held_codes)
    weights = inverse_vol_weights(target_codes, price_data, as_of, max_weight=weight_cap(top_k))
    score_of = dict(zip(ranked["코드"], ranked["모멘텀점수"]))
    rank_of = {c: i + 1 for i, c in enumerate(ranked["코드"])}

    regime = regime or {}
    rows = []
    for code in target_codes:
        if code not in weights:
            continue
        exposure = regime.get(markets.get(code, ""), (1.0, ""))[0]
        df = price_data[code]
        price = float(df["Close"].iloc[-1])
        atr = compute_atr(df).iloc[-1]
        stop = price - ATR_INITIAL_MULT * float(atr) if pd.notna(atr) else None
        rows.append({
            "순위": rank_of[code], "종목코드": code, "종목명": names.get(code, code),
            "시장": markets.get(code, ""), "모멘텀(%)": round(score_of[code] * 100, 1),
            "목표비중(%)": round(weights[code] * exposure * 100, 1), "현재가": price,
            "손절참고가": round(stop) if stop is not None else None,
            "상태": STATUS_HELD if code in held_codes else STATUS_NOT_HELD,
        })
    table = pd.DataFrame(rows).sort_values("순위").reset_index(drop=True)
    return table, as_of


def dropped_holdings(
    table: pd.DataFrame, held_codes: set[str], names: dict[str, str],
) -> list[tuple[str, str]]:
    """보유 중인데 이번 목표에서 빠진 종목 (코드, 종목명) — 월말에 정리 대상 후보."""
    target = set(table["종목코드"]) if not table.empty else set()
    return [(c, names.get(c, c)) for c in sorted(held_codes - target)]
