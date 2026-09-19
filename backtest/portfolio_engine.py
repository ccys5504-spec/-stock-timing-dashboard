"""포트폴리오 리밸런싱 백테스트 엔진 — PROMPT_V2.md Phase A1.

`backtest/engine.py`(v1)는 종목 1개, 신호 1개짜리 진입/청산 엔진이고 그대로
둔다. 이 엔진은 v2(상대강도 포트폴리오) 전략 전용으로 별개다.

체결 원칙은 v1과 동일하게 유지한다 — 그날 종가까지의 데이터로 내린 판단은
**다음 거래일 시가**에만 체결한다(리밸런싱 매매도, ATR 손절도 이 원칙을
따른다: 손절 기준선은 전일 종가까지 확정된 값이고 당일 저가로 이탈을
판정한다).

리밸런싱: 매월 마지막 거래일 종가까지의 데이터로 순위/비중을 계산하고, 다음
거래일 시가에 목표 비중대로 매매한다. 리밸런싱 사이에는 매일 ATR 손절/
트레일링 스탑만 확인한다(순위 재계산 없음). 손절로 청산된 포지션은 다음
리밸런싱까지 재진입하지 않는다(현금 보유) — 잦은 재진입으로 회전율이 튀는
것을 막기 위함(PROMPT_V2.md A1).

2026-09-16 추가 (회전율/비용 민감도 + 박스권 대응, PROMPT_V2.md Phase B 일부):
- **순위 이력(hysteresis)**: 처음엔 "상위 K에서 한 칸이라도 빠지면 곧바로
  매도"였는데, 이러면 순위가 엎치락뒤치락하는 것만으로도 불필요한 매매가
  계속 발생한다. 이제는 이미 보유 중인 종목은 순위가 컷오프(top_k × RANK_CUTOFF_MULTIPLE, 상위 K의
  2배) 밖으로 완전히 밀려나야 매도 대상이 되고, 빈 자리만 새 상위권 종목으로
  채운다.
- **소액 리밸런싱 생략**: 목표 비중과 현재 비중의 차이가 포트폴리오 가치의
  MIN_TRADE_THRESHOLD보다 작으면(이미 보유 중인 종목의 미세 조정에 한함 —
  신규 진입/완전 청산은 항상 실행됨) 거래를 생략한다. 매번 소수점 단위까지
  정확히 맞추려다 자잘한 거래비용만 쌓이는 걸 막기 위함.
- **추세강도(ADX) 기반 노출 축소**: 기존 장세 필터(A5)는 지수가 200일선
  위/아래인지만 본다. 그런데 2016-2019 같은 박스권은 지수가 200일선 근처를
  오르내리기만 할 뿐 "위/아래"가 자주 뒤집혀서 이 필터만으로는 못 걸러진다.
  Wilder의 ADX(추세 강도, 방향과 무관)가 20 미만(그가 제시한 "무추세" 경험적
  기준 — 데이터에 맞춰 고른 값이 아님)이면 노출 비중에 추가로 0.5배를 곱한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signals.indicators import ADX_NO_TREND_THRESHOLD, compute_adx, compute_atr
from strategy.momentum import rank_universe

# 매수/매도 각각에 절반씩 적용 — 왕복 기준 backtest/engine.py의
# ROUND_TRIP_COST(0.3%)와 같은 가정. run_portfolio_backtest()의
# round_trip_cost 인자로 실행 시점에 덮어쓸 수 있다(비용 스트레스 테스트용).
ROUND_TRIP_COST = 0.003

# 2026-09-20: 지수 200일선 장세필터를 기본으로 끈다(사용자 결정: 방어는 ATR 손절에 맡기고
# 수익 제고 우선). scripts/portfolio_ablation.py로 잰 결과 — 필터를 끄면 전체 수익률이
# 상위100/15종목 +357%→+618%, 상위200/15종목 +319%→+631%, 100/10종목 +196%→+349%,
# 100/20종목 +322%→+493%로 일관되게 오르고, 전체 MDD는 같거나 최대 6%p 악화된다.
# 대가: 2016~21 구간의 낙폭이 3~9%p 커지고 박스권(2016-18) 수익은 일부 설정에서 낮아진다.
# ⚠️ 종목군이 '오늘의 생존 종목'이라 하락 후 반등한 종목만 남아 있어, 필터의 방어
# 효과가 과소평가됐을 가능성이 있다. _regime_multiplier()는 그대로 두었다.
USE_REGIME_FILTER = False

TOP_K = 15
# 이미 보유 중인 종목은 순위가 (보유 종목 수 top_k) × 이 배수 밖으로 밀려나야 매도
# 대상이 된다 — 흔히 쓰는 "목표치의 2배" 버퍼 관행이며 데이터에 맞춰 고른 값이
# 아니다. 2026-09-20 수정: 예전엔 `RANK_CUTOFF = TOP_K * 2`(=30) 고정 상수를 썼기
# 때문에 run_portfolio_backtest(top_k=...)로 보유 종목 수를 바꿔도 컷오프가 30에
# 그대로여서 설계(top_k의 2배)와 다르게 돌았다. 지금은 top_k에서 매번 계산한다.
# (기본 top_k=15에서는 컷오프가 30으로 예전과 똑같아 문서에 기록된 결과는 그대로다.)
RANK_CUTOFF_MULTIPLE = 2
# 포트폴리오 가치 대비 이보다 작은 리밸런싱 조정은 생략한다(기존 보유 종목의
# 비중 미세조정에 한함 — 신규 진입/완전 청산에는 적용 안 됨).
MIN_TRADE_THRESHOLD = 0.015
MAX_WEIGHT_PER_STOCK = 0.12
ATR_PERIOD = 14
ATR_INITIAL_MULT = 2.5
ATR_TRAIL_MULT = 3.5
VOL_LOOKBACK_DAYS = 60
REGIME_MA_PERIOD = 200
REGIME_SLOPE_LOOKBACK = 20
ADX_DAMPEN_MULT = 0.5  # ADX가 무추세를 가리킬 때 노출 비중에 추가로 곱하는 배율
TRADING_DAYS_PER_YEAR = 252


@dataclass
class PortfolioTrade:
    code: str
    buy_date: pd.Timestamp
    buy_price: float  # 거래량가중 평균 매수가(비용 반영 전 명목가) — 보고용
    sell_date: pd.Timestamp | None = None
    sell_price: float | None = None
    stopped_out: bool = False  # 리밸런싱 제외가 아니라 ATR 손절로 청산됐는지

    @property
    def return_pct(self) -> float | None:
        if self.sell_price is None:
            return None
        return self.sell_price / self.buy_price - 1


@dataclass
class PortfolioBacktestResult:
    equity_curve: pd.Series
    trades: list[PortfolioTrade] = field(default_factory=list)
    # 리밸런싱마다 {날짜, 종목코드: 비중} — 어떤 시점에 뭘 얼마나 들고 있었는지 기록
    holdings_log: list[dict] = field(default_factory=list)

    @property
    def total_return_pct(self) -> float:
        return self.equity_curve.iloc[-1] / self.equity_curve.iloc[0] - 1

    @property
    def cagr(self) -> float | None:
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        if years <= 0:
            return None
        return (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) ** (1 / years) - 1

    @property
    def mdd_pct(self) -> float:
        cummax = self.equity_curve.cummax()
        return (self.equity_curve / cummax - 1).min()

    @property
    def daily_returns(self) -> pd.Series:
        return self.equity_curve.pct_change().dropna()

    @property
    def sharpe_ratio(self) -> float | None:
        r = self.daily_returns
        if len(r) < 2 or r.std() == 0:
            return None
        return float(r.mean() / r.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

    @property
    def num_trades(self) -> int:
        return len([t for t in self.trades if t.sell_price is not None])

    @property
    def win_rate_pct(self) -> float | None:
        closed = [t.return_pct for t in self.trades if t.return_pct is not None]
        if not closed:
            return None
        return sum(1 for r in closed if r > 0) / len(closed)

    @property
    def num_stopped_out(self) -> int:
        return sum(1 for t in self.trades if t.stopped_out)


def _regime_multiplier(index_df_upto: pd.DataFrame) -> float:
    """PROMPT_V2.md A5: 지수 200일선/기울기로 총 노출 비중 배율을 정한다."""
    close = index_df_upto["Close"]
    ma = close.rolling(REGIME_MA_PERIOD).mean().dropna()
    if ma.empty:
        return 1.0  # 데이터 부족(백테스트 초반) — 중립적으로 100%
    latest_close = close.iloc[-1]
    latest_ma = ma.iloc[-1]
    ref_idx = -1 - REGIME_SLOPE_LOOKBACK
    slope_ref = ma.iloc[ref_idx] if len(ma) > REGIME_SLOPE_LOOKBACK else ma.iloc[0]
    slope_positive = latest_ma > slope_ref
    above = latest_close > latest_ma
    if above:
        return 1.00 if slope_positive else 0.75
    return 0.40 if slope_positive else 0.20


def _trend_multiplier(index_df_upto: pd.DataFrame) -> float:
    """ADX가 "뚜렷한 추세가 없다"고 말하면 노출을 추가로 줄인다(2026-09-16
    추가). _regime_multiplier()는 가격이 200일선 위/아래인지만 보는데, 박스권
    에서는 그 위/아래가 자주 뒤집혀서 이 필터만으로는 놓치는 경우가 있다."""
    adx = compute_adx(index_df_upto).dropna()
    if adx.empty:
        return 1.0
    return ADX_DAMPEN_MULT if adx.iloc[-1] < ADX_NO_TREND_THRESHOLD else 1.0


def _select_with_hysteresis(
    ranked: pd.DataFrame, top_k: int, rank_cutoff: int, currently_held: set[str],
) -> list[str]:
    """순위 이력(hysteresis)을 반영해서 이번 리밸런싱의 목표 종목 목록을 정한다.

    이미 보유 중인 종목은 순위가 rank_cutoff 밖으로 밀려나야 제외 대상이 되고,
    그렇게 비는 자리만 아직 안 담은 종목 중 순위가 가장 높은 것으로 채운다.
    매 리밸런싱마다 처음부터 다시 뽑으면(순위가 K 언저리에서 계속 엎치락뒤치락
    하는 종목마다) 불필요한 매매가 반복되는 걸 줄이기 위함이다.
    """
    if ranked.empty:
        return []
    codes_in_order = ranked["코드"].tolist()
    rank_of = {c: i for i, c in enumerate(codes_in_order)}

    kept = [c for c in codes_in_order if c in currently_held and rank_of[c] < rank_cutoff]
    kept = kept[:top_k]  # codes_in_order 순서로 이미 순위 정렬돼 있음
    remaining = top_k - len(kept)
    new_picks = [c for c in codes_in_order if c not in kept][:max(remaining, 0)]
    return kept + new_picks


def _inverse_vol_weights(
    codes: list[str], price_data: dict[str, pd.DataFrame], as_of: pd.Timestamp,
    max_weight: float = MAX_WEIGHT_PER_STOCK,
) -> dict[str, float]:
    """변동성 역가중(위험균등) 비중, 종목당 상한을 적용하고 초과분은 나머지에
    비례 재분배한다(PROMPT_V2.md A4)."""
    vols: dict[str, float] = {}
    for code in codes:
        df = price_data.get(code)
        if df is None:
            continue
        df_upto = df[df.index <= as_of]
        rets = df_upto["Close"].pct_change().dropna().tail(VOL_LOOKBACK_DAYS)
        if len(rets) < 10:
            continue
        sigma = rets.std()
        if pd.isna(sigma) or sigma <= 0:
            continue
        vols[code] = float(sigma)

    if not vols:
        return {}

    inv = {c: 1 / s for c, s in vols.items()}
    total = sum(inv.values())
    weights = {c: v / total for c, v in inv.items()}

    for _ in range(20):  # 상한 캡 + 재분배를 수렴할 때까지 반복
        over = {c: w for c, w in weights.items() if w > max_weight + 1e-9}
        if not over:
            break
        excess = sum(w - max_weight for w in over.values())
        for c in over:
            weights[c] = max_weight
        under = {c: w for c, w in weights.items() if w < max_weight - 1e-9}
        under_total = sum(under.values())
        if under_total <= 0:
            break
        for c in under:
            weights[c] += excess * (weights[c] / under_total)

    return weights


def _month_end_dates(calendar: pd.DatetimeIndex) -> set[pd.Timestamp]:
    s = pd.Series(calendar, index=calendar)
    return set(s.groupby(calendar.to_period("M")).max().tolist())


def run_portfolio_backtest(
    price_data: dict[str, pd.DataFrame],
    market_by_code: dict[str, str],
    index_data: dict[str, pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
    top_k: int = TOP_K,
    round_trip_cost: float = ROUND_TRIP_COST,
    use_regime: bool = USE_REGIME_FILTER,
    atr_initial_mult: float | None = ATR_INITIAL_MULT,
    atr_trail_mult: float | None = ATR_TRAIL_MULT,
    weighting: str = "inverse_vol",
    stop_on_close: bool = False,
) -> PortfolioBacktestResult:
    """월 1회 리밸런싱 포트폴리오 백테스트.

    price_data: {종목코드: OHLCV DataFrame} — 순위 계산용 워밍업(최소 13개월)
      포함해서 start 이전 데이터도 들어있어야 한다.
    market_by_code: {종목코드: "KOSPI" 또는 "KOSDAQ"} — 장세 필터에 어떤 지수를
      적용할지 결정.
    index_data: {"KOSPI": 지수 OHLCV, "KOSDAQ": 지수 OHLCV}. 거래일 달력도
      이 중 하나(KOSPI 기준)에서 가져온다.
    round_trip_cost: 왕복 거래비용 가정치(기본 0.3%). 회전율이 높은 이
      전략에서 비용 가정이 결과를 얼마나 바꾸는지 보는 비용 스트레스
      테스트(scripts/portfolio_cost_stress.py)에서 1.5배/2배로 올려서 쓴다.

    use_regime/atr_initial_mult/atr_trail_mult/weighting은 구성요소를 하나씩 끄거나
    바꿔서 각각이 수익에 얼마나 기여하는지 재는 실험용 스위치다(scripts/
    portfolio_ablation.py). 기본값은 지금 운영 중인 v2 그대로라서 인자를 안 주면
    결과가 달라지지 않는다. atr_*_mult에 None을 주면 그 손절을 쓰지 않는다.
    stop_on_close=True면 장중 저가가 손절선을 건드려도 바로 팔지 않고, 그날 '종가'가
    손절선 아래일 때만 다음 거래일 시가에 판다(장중 꼬리에 털리는 걸 줄이는 대신
    갭하락 손실은 더 받는다).
    """
    if weighting not in ("inverse_vol", "equal"):
        raise ValueError(f"weighting은 'inverse_vol' 또는 'equal'이어야 합니다: {weighting}")
    buy_cost = round_trip_cost / 2
    sell_cost = round_trip_cost / 2

    calendar = index_data["KOSPI"].index
    calendar = calendar[(calendar >= start) & (calendar <= end)]
    if len(calendar) == 0:
        raise ValueError("해당 기간에 거래일이 없습니다.")

    rebalance_dates = _month_end_dates(calendar)

    # ATR은 종목당 한 번만 계산해서 재사용한다(호출마다 다시 계산하면 느림).
    # 이 캐시는 run_portfolio_backtest 호출마다 새로 만들어지는 지역 변수라,
    # 같은 종목코드라도 다른 price_data(예: 워크포워드의 다른 폴드)로 다시
    # 호출하면 새로 계산된다 — 모듈 전역 캐시로 두면 폴드 간에 값이 섞이는
    # 버그가 생기므로 반드시 지역 변수여야 한다.
    atr_cache: dict[str, pd.Series] = {}

    def _atr_at(code: str, date: pd.Timestamp) -> float:
        if code not in atr_cache:
            atr_cache[code] = compute_atr(price_data[code])
        series = atr_cache[code]
        if date in series.index and not pd.isna(series.loc[date]):
            return float(series.loc[date])
        close = price_data[code]["Close"]
        available = close[close.index <= date]
        fallback_price = float(available.iloc[-1]) if not available.empty else 0.0
        return fallback_price * 0.05

    cash = 1.0
    # code -> {"shares", "avg_buy_price", "buy_date", "peak", "stop_price"}
    holdings: dict[str, dict] = {}
    trades: list[PortfolioTrade] = []
    equity = []
    holdings_log = []

    pending_targets: dict[str, float] | None = None  # 다음 거래일 시가에 반영할 목표 비중
    pending_exits: set[str] = set()  # stop_on_close: 어제 종가가 손절선 아래여서 오늘 시가에 팔 종목

    # 세 함수 모두 0 이하 값은 None으로 취급한다 — 데이터 품질 문제(거래정지일에
    # 0으로 채워진 시가/저가 등)로 가격이 0 이하로 들어오면, 그걸 실제 가격으로
    # 쓰면 0으로 나누는 등 계산이 깨진다. None으로 처리하면 호출부에서 "오늘은
    # 이 종목 데이터가 없는 것"과 동일하게 안전히 건너뛴다.
    def _last_price(code: str, date: pd.Timestamp) -> float | None:
        df = price_data.get(code)
        if df is None or date not in df.index:
            return None
        price = float(df.loc[date, "Close"])
        return price if price > 0 else None

    def _open_price(code: str, date: pd.Timestamp) -> float | None:
        df = price_data.get(code)
        if df is None or date not in df.index:
            return None
        price = float(df.loc[date, "Open"])
        return price if price > 0 else None

    def _low_price(code: str, date: pd.Timestamp) -> float | None:
        df = price_data.get(code)
        if df is None or date not in df.index:
            return None
        price = float(df.loc[date, "Low"])
        return price if price > 0 else None

    def _close_trade(code: str, date: pd.Timestamp, price: float, stopped_out: bool) -> None:
        pos = holdings.pop(code)
        nonlocal cash
        cash += pos["shares"] * price * (1 - sell_cost)
        trades.append(PortfolioTrade(
            code=code, buy_date=pos["buy_date"], buy_price=pos["avg_buy_price"],
            sell_date=date, sell_price=price, stopped_out=stopped_out,
        ))

    for date in calendar:
        # 1) 전월 말 산정한 목표 비중을 오늘 시가에 반영 (룩어헤드 금지: 어제
        #    종가까지의 정보로 정한 목표를, 그 정보를 몰랐던 오늘 아침에 산다)
        if pending_targets is not None:
            portfolio_value = cash
            for code, pos in holdings.items():
                op = _open_price(code, date)
                if op is not None:
                    portfolio_value += pos["shares"] * op

            # 목표에 없는 기존 보유 종목은 전량 매도
            for code in list(holdings.keys()):
                if code not in pending_targets:
                    op = _open_price(code, date)
                    if op is not None:
                        _close_trade(code, date, op, stopped_out=False)
                    # 오늘 데이터가 없으면(거래정지 등) 다음날로 매도가 미뤄짐 — 단순화

            # 목표 비중대로 매수/조정
            for code, weight in pending_targets.items():
                op = _open_price(code, date)
                if op is None:
                    continue
                target_value = portfolio_value * weight
                current_shares = holdings.get(code, {}).get("shares", 0.0)
                current_value = current_shares * op
                delta = target_value - current_value

                # 이미 보유 중인 종목의 사소한 비중 미세조정은 생략한다(신규
                # 진입은 current_shares가 0이라 delta가 목표 비중 전체와 같으므로
                # 이 문턱에 걸리지 않는다 — 보통 top_k분의 1이 이 문턱보다 훨씬 큼).
                if current_shares > 0 and abs(delta) < portfolio_value * MIN_TRADE_THRESHOLD:
                    continue

                if delta > 0:  # 매수(신규 진입 포함)
                    spend = min(delta, cash)
                    if spend <= 0:
                        continue
                    bought_shares = spend * (1 - buy_cost) / op
                    cash -= spend
                    if code in holdings:
                        pos = holdings[code]
                        total_shares = pos["shares"] + bought_shares
                        pos["avg_buy_price"] = (
                            pos["avg_buy_price"] * pos["shares"] + op * bought_shares
                        ) / total_shares
                        pos["shares"] = total_shares
                    else:
                        holdings[code] = {
                            "shares": bought_shares, "avg_buy_price": op,
                            "buy_date": date, "peak": op,
                            "stop_price": (
                                op - atr_initial_mult * _atr_at(code, date)
                                if atr_initial_mult is not None else 0.0
                            ),
                        }
                elif delta < 0 and code in holdings:  # 비중 축소(부분 매도)
                    sell_shares = min(holdings[code]["shares"], -delta / op)
                    cash += sell_shares * op * (1 - sell_cost)
                    holdings[code]["shares"] -= sell_shares
                    if holdings[code]["shares"] <= 1e-9:
                        del holdings[code]

            pending_targets = None
            holdings_log.append({
                "날짜": date,
                "보유종목": {c: p["shares"] for c, p in holdings.items()},
            })

        # 2) ATR 손절/트레일링 체크 — 전일 종가까지 확정된 기준선 vs 당일 저가
        if stop_on_close:
            for code in list(pending_exits):
                if code not in holdings:
                    pending_exits.discard(code)
                    continue
                op = _open_price(code, date)
                if op is not None:  # 시가가 없으면(거래정지 등) 다음 거래일로 미룸
                    _close_trade(code, date, op, stopped_out=True)
                    pending_exits.discard(code)
        for code in list(holdings.keys()):
            if stop_on_close:
                break
            pos = holdings[code]
            low = _low_price(code, date)
            if low is None:
                continue
            if low <= pos["stop_price"]:
                op = _open_price(code, date) or low
                exit_price = min(op, pos["stop_price"])
                _close_trade(code, date, exit_price, stopped_out=True)

        # 3) 보유 지속 종목은 오늘 종가로 고점/ATR 기준선 갱신 (내일 기준선용).
        #    stop_price는 절대 내려가지 않게(max) 한다 — 진입 직후에는 진입가
        #    기준 초기 손절선(ATR_INITIAL_MULT)이 그대로 유지되다가, 가격이
        #    올라 트레일링 계산값(ATR_TRAIL_MULT)이 그걸 넘어서는 순간부터
        #    트레일링이 기준선을 이어받는다. 이렇게 안 하면 진입 당일 종가로
        #    바로 트레일링 계산을 덮어써서 초기 손절 폭(2.5×ATR)이 사실상
        #    하루 만에 트레일링 폭(3.5×ATR)으로 바뀌어버리는 문제가 있었다.
        for code, pos in holdings.items():
            close = _last_price(code, date)
            if close is None:
                continue
            if stop_on_close and close <= pos["stop_price"]:
                pending_exits.add(code)  # 어제까지 확정된 손절선 vs 오늘 종가
            pos["peak"] = max(pos["peak"], close)
            atr = _atr_at(code, date)
            if atr_trail_mult is not None:
                trailing_stop = pos["peak"] - atr_trail_mult * atr
                pos["stop_price"] = max(pos["stop_price"], trailing_stop)

        # 4) 오늘 종가 기준 평가금액
        value = cash
        for code, pos in holdings.items():
            close = _last_price(code, date)
            value += pos["shares"] * (close if close is not None else pos["avg_buy_price"])
        equity.append(value)

        # 5) 오늘이 월말이면, 오늘 종가까지의 정보로 다음 리밸런싱 목표를 정함
        if date in rebalance_dates:
            ranked = rank_universe(price_data, date)
            target_codes = _select_with_hysteresis(
                ranked, top_k, top_k * RANK_CUTOFF_MULTIPLE, set(holdings.keys())
            )
            if weighting == "equal":
                weights = {c: 1 / len(target_codes) for c in target_codes if c in price_data}
            else:
                weights = _inverse_vol_weights(target_codes, price_data, date)
            # 2026-09-16: ADX 무추세 필터(_trend_multiplier)를 여기서 뺐다.
            # 박스권(2016-2019) 방어에는 도움이 됐지만(샤프 0.05→0.42) 그
            # 대신 추세장에서도 노출을 깎아서 전체 기간 수익률이 288%→192%로
            # 낮아지고 단순보유보다도 낮아졌다. 사용자에게 "박스권 방어 vs
            # 전체 수익률" 중 뭘 우선할지 다시 물었더니 이번엔 수익률을
            # 선택해서, ADX 감쇠는 끄고 지수 200일선 기준 장세 필터만 남긴다.
            # 순위 이력(hysteresis)/소액 리밸런싱 생략은 회전율만 줄이고
            # 수익을 깎지 않아서 그대로 유지한다. compute_adx()/
            # _trend_multiplier() 자체는 지우지 않고 남겨뒀다 — 나중에 다시
            # 켜고 싶을 수도 있고, 이 결정의 맥락(왜 안 쓰는지)도 남겨야 해서.
            regime_mult = {
                market: _regime_multiplier(df[df.index <= date]) if use_regime else 1.0
                for market, df in index_data.items()
            }
            pending_targets = {
                code: w * regime_mult.get(market_by_code.get(code, "KOSPI"), 1.0)
                for code, w in weights.items()
            }

    equity_curve = pd.Series(equity, index=calendar)
    return PortfolioBacktestResult(equity_curve=equity_curve, trades=trades, holdings_log=holdings_log)
