"""이동평균 기간 / RSI 기준값 / 신호 임계값 / 거래량 필터 / 손절 비율 조합을
격자 탐색(grid search)해서 상승장(최근 3년)과 하락장(2022년) 양쪽에서 두루
괜찮은 조합을 찾는다.

⚠️ 주의: 과거 데이터에 대한 최적화(파라미터 피팅)는 과최적화(overfitting)
위험이 있다. 그래서 "상승장에서 가장 잘 번 조합"이 아니라 "상승장과 하락장
양쪽에서 최소한 어느 정도는 버티는(강건한) 조합"을 기준으로 고른다
— robust_score = min(상승장 수익률, 하락장 수익률).

vectorbt 등 벡터화 백테스트 라이브러리는 이 컴퓨터의 Windows 보안 정책이
새로 설치되는 numpy/scipy 네이티브 모듈을 차단해서 쓸 수 없었다. 그래서 순수
파이썬(+pandas)만으로 이전보다 훨씬 넓은 조합(이동평균 기간, RSI 기준값까지)을
탐색하도록 확장했다 — 지표 계산은 고유한 (MA/RSI 기간) 조합별로 한 번씩만
미리 계산해서 재사용하고, 그 위에서 임계값/거래량필터/손절만 바꿔가며 빠르게
반복한다.

사용법: python scripts/optimize.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _pyarrow_compat  # noqa: F401,E402  # pandas보다 반드시 먼저 임포트

import pandas as pd  # noqa: E402

from data.fetch import WATCHLIST, fetch_ohlcv, fetch_ohlcv_range
from signals.indicators import add_indicators, add_scores
from backtest.engine import run_backtest, buy_and_hold_return_pct

BEAR_START, BEAR_END = "2022-01-01", "2022-12-31"

# ---- 지표 계산 자체를 바꾸는 파라미터 (조합별로 한 번씩만 계산해서 재사용) ----
MA_COMBOS = [(10, 40), (20, 60), (20, 90), (30, 90), (10, 60)]  # (ma_short, ma_long)
RSI_COMBOS = [(25, 75), (30, 70), (35, 65)]  # (oversold, overbought)

# ---- 신호/백테스트 파라미터 ----
THRESHOLDS = [1, 2]
VOLUME_FILTERS = [False, True]
STOP_LOSSES = [None, 0.07, 0.10, 0.15]

# 상승장 기간(3년) 동안 최소 이 정도는 실제로 매매해야 "전략"으로 인정한다.
# 그렇지 않으면 "임계값을 너무 높여서 사실상 매매를 안 함 = 현금 보유"인
# 경우가 손실 회피 측면에서 점수만 높게 나오는 착시가 생긴다.
MIN_MEANINGFUL_TRADES = 5


def evaluate(df_ind_bull, df_ind_bear, ma_combo, rsi_combo, threshold, volume_filter, stop_loss):
    ma_short, ma_long = ma_combo
    rsi_os, rsi_ob = rsi_combo

    df_bull = add_scores(
        df_ind_bull, buy_threshold=threshold, sell_threshold=-threshold,
        volume_filter=volume_filter, rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
    )
    df_bear = add_scores(
        df_ind_bear, buy_threshold=threshold, sell_threshold=-threshold,
        volume_filter=volume_filter, rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
    )
    df_bear = df_bear[(df_bear.index >= pd.Timestamp(BEAR_START)) & (df_bear.index <= pd.Timestamp(BEAR_END))]

    r_bull = run_backtest(df_bull, stop_loss_pct=stop_loss)
    r_bear = run_backtest(df_bear, stop_loss_pct=stop_loss)

    return {
        "ma": f"{ma_short}/{ma_long}",
        "rsi": f"{rsi_os}/{rsi_ob}",
        "threshold": threshold,
        "volume_filter": volume_filter,
        "stop_loss": stop_loss,
        "bull_return": r_bull.total_return_pct,
        "bear_return": r_bear.total_return_pct,
        "bull_trades": r_bull.num_trades,
        "bear_trades": r_bear.num_trades,
        "robust_score": min(r_bull.total_return_pct, r_bear.total_return_pct),
        "avg_score": (r_bull.total_return_pct + r_bear.total_return_pct) / 2,
    }


def main() -> None:
    n_combos = (
        len(MA_COMBOS) * len(RSI_COMBOS) * len(THRESHOLDS)
        * len(VOLUME_FILTERS) * len(STOP_LOSSES)
    )
    print(f"종목당 {n_combos}개 조합 x 2개 기간(상승장/2022년) 탐색\n")

    for name, code in WATCHLIST.items():
        print(f"\n{'=' * 70}\n[{name} ({code})]\n{'=' * 70}")

        df_bull_raw = fetch_ohlcv(code, years=3)
        df_bear_raw = fetch_ohlcv_range(code, BEAR_START, BEAR_END)

        # 지표는 (ma_combo, rsi_combo) 조합별로 한 번만 계산해서 재사용한다
        # (RSI 기간 자체는 고정, oversold/overbought 기준값만 add_scores에서 바뀌므로
        # 실제로는 ma_combo 개수만큼만 계산하면 된다).
        indicator_cache_bull = {}
        indicator_cache_bear = {}
        for ma_short, ma_long in MA_COMBOS:
            indicator_cache_bull[(ma_short, ma_long)] = add_indicators(
                df_bull_raw, ma_short=ma_short, ma_long=ma_long
            ).dropna(subset=["MA_LONG"])
            indicator_cache_bear[(ma_short, ma_long)] = add_indicators(
                df_bear_raw, ma_short=ma_short, ma_long=ma_long
            ).dropna(subset=["MA_LONG"])

        default_ind_bull = indicator_cache_bull[MA_COMBOS[0]]
        bull_bh = buy_and_hold_return_pct(default_ind_bull)
        bear_period = indicator_cache_bear[MA_COMBOS[0]]
        bear_period = bear_period[
            (bear_period.index >= pd.Timestamp(BEAR_START)) & (bear_period.index <= pd.Timestamp(BEAR_END))
        ]
        bear_bh = buy_and_hold_return_pct(bear_period)
        print(f"  (참고) 단순보유: 상승장 {bull_bh:+.1%} / 2022년 {bear_bh:+.1%}\n")

        results = []
        for ma_combo, rsi_combo, threshold, vfilter, sl in itertools.product(
            MA_COMBOS, RSI_COMBOS, THRESHOLDS, VOLUME_FILTERS, STOP_LOSSES
        ):
            df_ind_bull = indicator_cache_bull[ma_combo]
            df_ind_bear = indicator_cache_bear[ma_combo]
            results.append(
                evaluate(df_ind_bull, df_ind_bear, ma_combo, rsi_combo, threshold, vfilter, sl)
            )

        def show(rows, label):
            print(f"  [{label}]")
            print(
                f"  {'MA':>7} {'RSI':>7} {'임계값':>5} {'거래량':>6} {'손절':>6} | "
                f"{'상승장':>8} {'2022년':>8} | {'거래수(상/하)':>12}"
            )
            for r in rows[:8]:
                sl_label = "없음" if r["stop_loss"] is None else f"{r['stop_loss']:.0%}"
                print(
                    f"  {r['ma']:>7} {r['rsi']:>7} ±{r['threshold']:>4} "
                    f"{str(r['volume_filter']):>6} {sl_label:>6} | "
                    f"{r['bull_return']:>+7.1%} {r['bear_return']:>+7.1%} | "
                    f"{r['bull_trades']:>5}회/{r['bear_trades']:<5}회"
                )
            print()

        meaningful = [r for r in results if r["bull_trades"] >= MIN_MEANINGFUL_TRADES]
        meaningful.sort(key=lambda r: r["robust_score"], reverse=True)
        if meaningful:
            show(meaningful, f"실제로 매매하는 조합만(상승장 {MIN_MEANINGFUL_TRADES}회 이상) 중 강건성 상위 8개")
        else:
            print(f"  ⚠️ 상승장 기준 {MIN_MEANINGFUL_TRADES}회 이상 매매하는 조합이 없습니다.\n")


if __name__ == "__main__":
    main()
