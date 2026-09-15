"""v2 포트폴리오 전략의 거래비용 민감도 검증 — PROMPT_V2.md Phase B.

이 전략은 월 1회 리밸런싱마다 여러 종목을 동시에 갈아타서 v1(3종목 신호
전략)보다 회전율이 훨씬 높다(scripts/portfolio_regime_check.py 결과 기준
연 100회 이상). 회전율이 높을수록 "거래비용을 실제보다 낙관적으로 가정했다면
결과가 순식간에 뒤집힐 수 있다"는 위험이 커지므로, 왕복 비용 가정치를
1배(0.3%, 기본) / 1.5배(0.45%) / 2배(0.6%)로 올려가며 같은 전략을 다시
돌려본다.

사용법: python scripts/portfolio_cost_stress.py [유니버스크기] [보유종목수] [시작일] [종료일]
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _pyarrow_compat  # noqa: F401,E402

import pandas as pd  # noqa: E402

from backtest.engine import buy_and_hold_return_pct  # noqa: E402
from backtest.portfolio_engine import ROUND_TRIP_COST, run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range, get_universe  # noqa: E402

WARMUP_DAYS = 450
MAX_WORKERS = 12
COST_MULTIPLIERS = [1.0, 1.5, 2.0]


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

    def _fetch(code: str):
        return code, fetch_ohlcv_range(code, fetch_start, end, warmup_days=WARMUP_DAYS)

    price_data: dict[str, pd.DataFrame] = {}
    market_by_code: dict[str, str] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch, row["Code"]): row["Market"] for _, row in universe.iterrows()}
        for future in as_completed(futures):
            market = futures[future]
            try:
                code, df = future.result()
            except Exception:  # noqa: BLE001
                continue
            price_data[code] = df
            market_by_code[code] = market
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(universe)} 완료")

    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", fetch_start, end, warmup_days=WARMUP_DAYS),
        "KOSDAQ": fetch_ohlcv_range("KQ11", fetch_start, end, warmup_days=WARMUP_DAYS),
    }
    print(f"{len(price_data)}개 종목 확보 완료.\n")

    kospi_period = index_data["KOSPI"]
    kospi_period = kospi_period[
        (kospi_period.index >= pd.Timestamp(start)) & (kospi_period.index <= pd.Timestamp(end))
    ]
    kospi_bh = buy_and_hold_return_pct(kospi_period) if not kospi_period.empty else None

    rows = []
    for mult in COST_MULTIPLIERS:
        cost = ROUND_TRIP_COST * mult
        result = run_portfolio_backtest(
            price_data, market_by_code, index_data,
            start=pd.Timestamp(start), end=pd.Timestamp(end), top_k=top_k,
            round_trip_cost=cost,
        )
        rows.append({
            "배수": mult, "왕복비용": cost,
            "누적수익률": result.total_return_pct, "CAGR": result.cagr,
            "MDD": result.mdd_pct, "샤프": result.sharpe_ratio,
            "매매횟수": result.num_trades,
        })

    print(f"{'비용 가정':>10} {'누적수익률':>10} {'CAGR':>8} {'MDD':>8} {'샤프':>6} {'매매횟수':>7}")
    for r in rows:
        cagr_s = f"{r['CAGR']:+.1%}" if r["CAGR"] is not None else "—"
        sharpe_s = f"{r['샤프']:.2f}" if r["샤프"] is not None else "—"
        print(
            f"{r['왕복비용']:>9.2%} {r['누적수익률']:>+9.1%} {cagr_s:>8} "
            f"{r['MDD']:>+7.1%} {sharpe_s:>6} {r['매매횟수']:>6}회"
        )
    if kospi_bh is not None:
        print(f"\n(참고) 같은 기간 코스피 단순보유: {kospi_bh:+.1%}")

    base = rows[0]["누적수익률"]
    worst = rows[-1]["누적수익률"]
    print(
        f"\n비용을 {COST_MULTIPLIERS[0]:.0%}에서 {COST_MULTIPLIERS[-1]:.0%}로(왕복 "
        f"{ROUND_TRIP_COST:.1%}→{ROUND_TRIP_COST*COST_MULTIPLIERS[-1]:.1%}) 올리면 "
        f"누적수익률이 {base:+.1%}에서 {worst:+.1%}로 변합니다 — 이 차이가 클수록 "
        "이 전략이 실제 거래비용(슬리피지·호가 스프레드 포함)을 낙관적으로 가정했을 "
        "때 결과가 얼마나 취약한지를 보여줍니다."
    )


if __name__ == "__main__":
    main()
