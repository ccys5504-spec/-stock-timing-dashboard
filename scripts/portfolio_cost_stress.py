"""v2 포트폴리오 전략의 거래비용 민감도 검증 — PROMPT_V2.md Phase B.

⚠️ 이 스크립트의 종목군은 "오늘의 시총 상위 N개"라 생존편향이 있습니다(2026-09-20 확인: 같은 전략이 편향
종목군 +617.8% → 시점 기준 종목군 -1.5%). 편향 없는 검증은 scripts/portfolio_pit_backtest.py를 쓰세요.

이 전략은 월 1회 리밸런싱마다 여러 종목을 동시에 갈아타서 v1(3종목 신호
전략)보다 회전율이 훨씬 높다(연 100회 안팎). 회전율이 높을수록 "거래비용을
실제보다 낙관적으로 가정했다면 결과가 순식간에 뒤집힐 수 있다"는 위험이 커지므로,
왕복 비용 가정치를 0.3%(기본) / 0.45% / 0.6% / **1.0%**로 올려가며 같은 전략을
다시 돌려본다.

2026-09-20: 1.0% 단계를 추가했다(외부 점검서: 호가 스프레드·슬리피지·세금 비대칭까지
감안하면 최소 0.6~1.0%는 봐야 함). 비교 기준선에 같은 종목군 동일가중 보유도
함께 출력한다. 조회에 실패한 종목은 목록으로 출력된다.

사용법: python scripts/portfolio_cost_stress.py [유니버스크기] [보유종목수] [시작일] [종료일]
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pyarrow_compat  # noqa: F401,E402

import pandas as pd  # noqa: E402

from _common import WARMUP_DAYS, load_universe_prices, report_load  # noqa: E402
from backtest.benchmarks import equal_weight_equity, summarize_equity  # noqa: E402
from backtest.engine import buy_and_hold_return_pct  # noqa: E402
from backtest.portfolio_engine import ROUND_TRIP_COST, run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range, get_universe  # noqa: E402

# 왕복 비용 가정치(비율). 0.003이 기본 가정이고 나머지는 스트레스 단계.
COSTS = [ROUND_TRIP_COST, 0.0045, 0.006, 0.010]


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    start = sys.argv[3] if len(sys.argv) > 3 else "2016-01-01"
    end = sys.argv[4] if len(sys.argv) > 4 else pd.Timestamp.today().date().isoformat()
    fetch_start = (pd.Timestamp(start) - pd.DateOffset(months=7)).date().isoformat()

    print(f"=== v2 포트폴리오 전략 비용 스트레스 테스트: {start} ~ {end} ===")
    print(f"유니버스 상위 {top_n}개, 보유 {top_k}종목, 기본 왕복비용 {ROUND_TRIP_COST:.1%}\n")

    universe = get_universe(markets=("KOSPI", "KOSDAQ"), top_n=top_n)
    print(f"{len(universe)}개 종목 데이터 조회 중... (병렬)")
    price_data, market_by_code, failed = load_universe_prices(universe, fetch_start, end)
    report_load(len(universe), price_data, failed)

    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", fetch_start, end, warmup_days=WARMUP_DAYS),
        "KOSDAQ": fetch_ohlcv_range("KQ11", fetch_start, end, warmup_days=WARMUP_DAYS),
    }

    kospi_period = index_data["KOSPI"]
    kospi_period = kospi_period[
        (kospi_period.index >= pd.Timestamp(start)) & (kospi_period.index <= pd.Timestamp(end))
    ]
    kospi_bh = buy_and_hold_return_pct(kospi_period) if not kospi_period.empty else None
    ew_total, ew_mdd = summarize_equity(equity=equal_weight_equity(price_data, pd.Timestamp(start), pd.Timestamp(end)))

    rows = []
    for cost in COSTS:
        result = run_portfolio_backtest(
            price_data, market_by_code, index_data,
            start=pd.Timestamp(start), end=pd.Timestamp(end), top_k=top_k,
            round_trip_cost=cost,
        )
        rows.append({
            "왕복비용": cost,
            "누적수익률": result.total_return_pct, "CAGR": result.cagr,
            "MDD": result.mdd_pct, "샤프": result.sharpe_ratio,
            "매매횟수": result.num_trades,
        })

    print(f"{'왕복비용':>8} {'누적수익률':>10} {'CAGR':>8} {'MDD':>8} {'샤프':>6} {'매매횟수':>7}")
    for r in rows:
        cagr_s = f"{r['CAGR']:+.1%}" if r["CAGR"] is not None else "—"
        sharpe_s = f"{r['샤프']:.2f}" if r["샤프"] is not None else "—"
        print(
            f"{r['왕복비용']:>7.2%} {r['누적수익률']:>+9.1%} {cagr_s:>8} "
            f"{r['MDD']:>+7.1%} {sharpe_s:>6} {r['매매횟수']:>6}회"
        )

    print("\n[비교 기준선 — 같은 기간]")
    if kospi_bh is not None:
        print(f"  코스피 지수 단순보유          : {kospi_bh:+.1%}")
    print(f"  같은 종목군 동일가중 보유      : {ew_total:+.1%} (MDD {ew_mdd:.1%}, 비용·생존편향 미반영 상한선)")

    print(
        f"\n비용을 {COSTS[0]:.1%}에서 {COSTS[-1]:.1%}로 올리면 누적수익률이 "
        f"{rows[0]['누적수익률']:+.1%}에서 {rows[-1]['누적수익률']:+.1%}로 변합니다."
    )


if __name__ == "__main__":
    main()
