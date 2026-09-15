"""임의 기간(예: 하락장 포함 구간)으로 관심 종목들의 백테스트 요약을 출력한다.

사용법: python scripts/backtest_range_report.py <시작일> <종료일>
예)    python scripts/backtest_range_report.py 2021-01-01 2022-12-31
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _pyarrow_compat  # noqa: F401,E402  # pandas보다 반드시 먼저 임포트

import pandas as pd  # noqa: E402

from data.fetch import WATCHLIST, fetch_ohlcv_range
from signals.indicators import build_signals
from backtest.engine import run_backtest, buy_and_hold_return_pct


def main() -> None:
    if len(sys.argv) < 3:
        print("사용법: python scripts/backtest_range_report.py <시작일 YYYY-MM-DD> <종료일 YYYY-MM-DD>")
        sys.exit(1)

    start, end = sys.argv[1], sys.argv[2]
    print(f"=== {start} ~ {end} 백테스트 요약 (하락장 포함 여부 확인용) ===\n")

    for name, code in WATCHLIST.items():
        try:
            df = fetch_ohlcv_range(code, start, end)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] 데이터 조회 실패: {e}\n")
            continue

        df_sig = build_signals(df)
        df_sig = df_sig.dropna(subset=["MA_LONG"])
        # 워밍업 구간을 뺀 실제 요청 구간만 사용
        df_period = df_sig[(df_sig.index >= pd.Timestamp(start)) & (df_sig.index <= pd.Timestamp(end))]

        if df_period.empty:
            print(f"[{name}] 해당 구간에 데이터가 없습니다.\n")
            continue

        result = run_backtest(df_period)
        bh_return = buy_and_hold_return_pct(df_period)

        print(f"[{name} ({code})]")
        print(f"  기간           : {df_period.index.min().date()} ~ {df_period.index.max().date()}")
        print(f"  전략 누적수익률 : {result.total_return_pct:+.1%}")
        print(f"  단순보유 수익률 : {bh_return:+.1%}")
        print(f"  최대낙폭(MDD)  : {result.mdd_pct:.1%}")
        print(f"  총 매매 횟수    : {result.num_trades}회")
        if result.avg_holding_days is not None:
            print(f"  평균 보유일수   : {result.avg_holding_days:.0f}일")
        if result.num_trades > 0:
            total_days = (df_period.index.max() - df_period.index.min()).days
            avg_gap = total_days / result.num_trades
            print(f"  평균 매매 주기  : 약 {avg_gap:.0f}일에 1회 (매수+매도 한 쌍 기준)")
        if result.win_rate_pct is not None:
            print(f"  승률           : {result.win_rate_pct:.0%}")
        if result.sharpe_ratio is not None:
            print(f"  샤프 비율       : {result.sharpe_ratio:.2f}")
        if result.sortino_ratio is not None:
            print(f"  소르티노 비율   : {result.sortino_ratio:.2f}")
        if result.exposure_pct is not None:
            print(f"  시장 노출도     : {result.exposure_pct:.0%}")
        outperform = "전략이 방어적" if result.total_return_pct > bh_return else "단순보유가 더 나음"
        print(f"  비교 결과       : {outperform} (전략 {result.total_return_pct:+.1%} vs 보유 {bh_return:+.1%})")
        print()


if __name__ == "__main__":
    main()
