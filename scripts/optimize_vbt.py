"""vectorbt 기반 대규모 파라미터 최적화.

⚠️ 이 스크립트는 Windows에서 직접 실행할 수 없다. 이 컴퓨터의 Windows
애플리케이션 제어 정책이 numpy/scipy의 새 네이티브 모듈을 차단해서 vectorbt가
동작하지 않기 때문이다 (README.md 참고). 대신 WSL2(Ubuntu) 안에 만들어 둔
전용 가상환경에서 실행한다:

    wsl -d Ubuntu -u root -- /root/vbt-venv/bin/python \
        "/mnt/c/Users/<사용자명>/Desktop/AI 저작물/주식매매 타이밍/scripts/optimize_vbt.py"

이동평균 단기/장기 기간 조합(기본 6x7=42개) x 손절 비율(7개) = 총 294개 조합을
종목당 단 몇 번의 벡터화 연산으로 백테스트한다. scripts/optimize.py(순수
파이썬, 240개 조합, 종목당 약 10초)보다 조합 수는 비슷하거나 더 많으면서
훨씬 빠르다 — 동일한 컴퓨터에서 여러 배로 더 큰 그리드도 감당할 수 있다는
뜻이고, 그만큼 더 다양한 가설을 검증할 수 있다.

⚠️ 여기서는 이동평균 골든/데드크로스 단일 전략만 다룬다 (RSI/MACD/볼린저밴드
조합 점수 로직 전체를 vectorbt로 옮기는 건 더 큰 작업이라 범위에서 제외했다).
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import vectorbt as vbt  # noqa: E402

from data.fetch import WATCHLIST, fetch_ohlcv, fetch_ohlcv_range  # noqa: E402

BEAR_START, BEAR_END = "2022-01-01", "2022-12-31"

MA_SHORT_OPTS = [5, 10, 15, 20, 25, 30]
MA_LONG_OPTS = [40, 50, 60, 70, 80, 90, 100]
STOP_LOSS_OPTS = [np.nan, 0.05, 0.07, 0.10, 0.12, 0.15, 0.20]  # nan = 손절 없음
FEES = 0.0015  # 편도 0.15% (왕복 약 0.3%, 우리 backtest/engine.py의 ROUND_TRIP_COST와 동일 가정)

MIN_MEANINGFUL_TRADES = 3


def build_entries_exits(close: pd.Series):
    """(fast_window, slow_window) 전 조합에 대한 골든/데드크로스 entries/exits.

    vbt.MA.run()이 만드는 fast/slow 결과는 각자 컬럼 개수가 달라서(6개 vs 7개)
    IndicatorFactory의 ma_crossed_above/below가 바로 비교하지 못한다
    (컬럼 수가 다르면 위치 정렬이 안 됨). 그래서 두 이평선을 (fast, slow)
    MultiIndex 조합으로 직접 펼쳐서 비교한다.
    """
    fast_ma = vbt.MA.run(close, window=MA_SHORT_OPTS, short_name="fast").ma
    slow_ma = vbt.MA.run(close, window=MA_LONG_OPTS, short_name="slow").ma

    combos = list(itertools.product(MA_SHORT_OPTS, MA_LONG_OPTS))
    fast_expanded = pd.concat([fast_ma[f] for f, s in combos], axis=1)
    slow_expanded = pd.concat([slow_ma[s] for f, s in combos], axis=1)
    columns = pd.MultiIndex.from_tuples(combos, names=["fast_window", "slow_window"])
    fast_expanded.columns = columns
    slow_expanded.columns = columns

    diff = fast_expanded - slow_expanded
    prev_diff = diff.shift(1)
    entries = (prev_diff <= 0) & (diff > 0)
    exits = (prev_diff >= 0) & (diff < 0)
    return entries, exits


def run_grid(close: pd.Series) -> pd.DataFrame:
    """(fast_window, slow_window, stop_loss)별 total_return/mdd/거래수를 담은
    DataFrame을 반환한다."""
    entries, exits = build_entries_exits(close)

    rows = []
    for sl in STOP_LOSS_OPTS:
        sl_arg = None if np.isnan(sl) else sl
        pf = vbt.Portfolio.from_signals(
            close, entries, exits, sl_stop=sl_arg, fees=FEES, freq="1D",
        )
        total_return = pf.total_return()
        mdd = pf.max_drawdown()
        n_trades = pf.trades.count()
        for (fast_w, slow_w) in total_return.index:
            rows.append({
                "ma": f"{fast_w}/{slow_w}",
                "stop_loss": sl,
                "total_return": total_return[(fast_w, slow_w)],
                "mdd": mdd[(fast_w, slow_w)],
                "trades": int(n_trades[(fast_w, slow_w)]),
            })
    return pd.DataFrame(rows)


def main() -> None:
    n_combos = len(MA_SHORT_OPTS) * len(MA_LONG_OPTS) * len(STOP_LOSS_OPTS)
    print(f"종목당 {n_combos}개 조합 (MA {len(MA_SHORT_OPTS)}x{len(MA_LONG_OPTS)} "
          f"x 손절 {len(STOP_LOSS_OPTS)}개) x 2개 기간(상승장/2022년) 탐색\n")

    for name, code in WATCHLIST.items():
        print(f"\n{'=' * 70}\n[{name} ({code})]\n{'=' * 70}")

        bull_close = fetch_ohlcv(code, years=3)["Close"]
        bear_raw = fetch_ohlcv_range(code, BEAR_START, BEAR_END, warmup_days=200)["Close"]
        bear_close_full = bear_raw
        bear_close = bear_raw.loc[BEAR_START:BEAR_END]

        bull_bh = bull_close.iloc[-1] / bull_close.iloc[0] - 1
        bear_bh = bear_close.iloc[-1] / bear_close.iloc[0] - 1
        print(f"  (참고) 단순보유: 상승장 {bull_bh:+.1%} / 2022년 {bear_bh:+.1%}\n")

        bull_df = run_grid(bull_close).rename(
            columns={"total_return": "bull_return", "mdd": "bull_mdd", "trades": "bull_trades"}
        )

        # 하락장은 entries/exits를 전체(워밍업 포함) 구간에서 계산한 뒤 2022년만 슬라이싱
        entries_full, exits_full = build_entries_exits(bear_close_full)
        entries_bear = entries_full.loc[BEAR_START:BEAR_END]
        exits_bear = exits_full.loc[BEAR_START:BEAR_END]
        bear_rows = []
        for sl in STOP_LOSS_OPTS:
            sl_arg = None if np.isnan(sl) else sl
            pf = vbt.Portfolio.from_signals(
                bear_close, entries_bear, exits_bear, sl_stop=sl_arg, fees=FEES, freq="1D",
            )
            total_return = pf.total_return()
            n_trades = pf.trades.count()
            for (fast_w, slow_w) in total_return.index:
                bear_rows.append({
                    "ma": f"{fast_w}/{slow_w}",
                    "stop_loss": sl,
                    "bear_return": total_return[(fast_w, slow_w)],
                    "bear_trades": int(n_trades[(fast_w, slow_w)]),
                })
        bear_df = pd.DataFrame(bear_rows)

        merged = bull_df.merge(bear_df, on=["ma", "stop_loss"])
        merged["robust_score"] = merged[["bull_return", "bear_return"]].min(axis=1)

        meaningful = merged[merged["bull_trades"] >= MIN_MEANINGFUL_TRADES].copy()
        meaningful = meaningful.sort_values("robust_score", ascending=False)

        print(f"  [강건성 상위 8개] (전체 {len(merged)}개 조합 중, 상승장 최소 "
              f"{MIN_MEANINGFUL_TRADES}회 이상 거래한 것만)")
        print(f"  {'MA':>8} {'손절':>6} | {'상승장':>8} {'2022년':>8} | {'거래(상/하)':>10}")
        for _, r in meaningful.head(8).iterrows():
            sl_label = "없음" if pd.isna(r["stop_loss"]) else f"{r['stop_loss']:.0%}"
            print(
                f"  {r['ma']:>8} {sl_label:>6} | {r['bull_return']:>+7.1%} "
                f"{r['bear_return']:>+7.1%} | {r['bull_trades']:>4}회/{r['bear_trades']:<4}회"
            )
        print()


if __name__ == "__main__":
    main()
