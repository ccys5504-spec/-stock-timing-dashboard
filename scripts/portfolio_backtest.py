"""v2 포트폴리오(상대강도) 전략 백테스트 CLI — PROMPT_V2.md Phase A.

관심종목 3개짜리 v1 신호 전략과는 별개로, 시가총액 상위 N개 종목 중 모멘텀
(상대강도) 상위 종목을 골라 월 1회 리밸런싱하는 v2 전략을 검증한다.

사용법: python scripts/portfolio_backtest.py [시작일] [종료일] [유니버스크기] [보유종목수]
예)    python scripts/portfolio_backtest.py 2019-01-01 2025-12-31 200 15
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pyarrow_compat  # noqa: F401,E402  # pandas보다 반드시 먼저 임포트

import pandas as pd  # noqa: E402

from _common import WARMUP_DAYS, load_universe_prices, report_load  # noqa: E402
from backtest.benchmarks import equal_weight_equity, summarize_equity  # noqa: E402
from backtest.engine import buy_and_hold_return_pct  # noqa: E402
from backtest.portfolio_engine import run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range, get_universe  # noqa: E402


def main() -> None:
    start = sys.argv[1] if len(sys.argv) > 1 else "2019-01-01"
    end = sys.argv[2] if len(sys.argv) > 2 else pd.Timestamp.today().date().isoformat()
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    top_k = int(sys.argv[4]) if len(sys.argv) > 4 else 15

    print(
        f"=== v2 포트폴리오 전략 백테스트: {start} ~ {end}, "
        f"유니버스 상위 {top_n}개 중 {top_k}종목 보유 ===\n"
    )

    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", start, end, warmup_days=WARMUP_DAYS),
        "KOSDAQ": fetch_ohlcv_range("KQ11", start, end, warmup_days=WARMUP_DAYS),
    }

    universe = get_universe(markets=("KOSPI", "KOSDAQ"), top_n=top_n)
    print(f"{len(universe)}개 종목 데이터 조회 중... (병렬, 수 분 소요될 수 있습니다)")
    price_data, market_by_code, failed = load_universe_prices(universe, start, end)
    report_load(len(universe), price_data, failed)

    result = run_portfolio_backtest(
        price_data, market_by_code, index_data,
        start=pd.Timestamp(start), end=pd.Timestamp(end), top_k=top_k,
    )

    print(f"기간              : {result.equity_curve.index.min().date()} ~ {result.equity_curve.index.max().date()}")
    print(f"누적수익률         : {result.total_return_pct:+.1%}")
    cagr = result.cagr
    print(f"CAGR             : {cagr:+.1%}" if cagr is not None else "CAGR             : 계산불가")
    print(f"최대낙폭(MDD)     : {result.mdd_pct:.1%}")
    sharpe = result.sharpe_ratio
    print(f"샤프 비율         : {sharpe:.2f}" if sharpe is not None else "샤프 비율         : 계산불가")
    print(f"총 매매 횟수(청산) : {result.num_trades}회")
    wr = result.win_rate_pct
    print(f"승률             : {wr:.0%}" if wr is not None else "승률             : —")
    print(f"ATR 손절 청산     : {result.num_stopped_out}회")

    kospi_period = index_data["KOSPI"]
    kospi_period = kospi_period[
        (kospi_period.index >= pd.Timestamp(start)) & (kospi_period.index <= pd.Timestamp(end))
    ]
    print(f"\n[비교 기준선 — 같은 기간]")
    print(f"  코스피 지수 단순보유        : {buy_and_hold_return_pct(kospi_period):+.1%}")
    ew_total, ew_mdd = summarize_equity(equal_weight_equity(price_data, pd.Timestamp(start), pd.Timestamp(end)))
    print(f"  같은 종목군 동일가중 보유    : {ew_total:+.1%} (MDD {ew_mdd:.1%}, 생존편향 섞인 상한선)")


if __name__ == "__main__":
    main()
