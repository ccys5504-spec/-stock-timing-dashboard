"""v2 포트폴리오 전략을 서로 다른 시장 국면 여러 구간에서 반복 검증한다.

파라미터(룩백 개월, ATR 배수 등)를 그리드 탐색으로 데이터에 맞추는 과정이
없으므로(문헌 표준값 고정 — PROMPT_V2.md 검증 방법론 1번), 전통적인 의미의
훈련/검증 워크포워드는 필요 없다. 대신 "우연히 한 구간에서만 잘 된 것"인지
확인하기 위해 서로 다른 국면 여러 구간에서 같은 전략을 그대로 돌려본다
(PROMPT_V2.md 검증 방법론 2번).

사용법: python scripts/portfolio_regime_check.py [유니버스크기] [보유종목수]
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
from backtest.portfolio_engine import run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range, get_universe  # noqa: E402

FETCH_START = "2015-06-01"  # 2016-01-01 구간 테스트에 필요한 13개월 워밍업 확보
WARMUP_DAYS = 450
MAX_WORKERS = 12

WINDOWS = [
    ("2016-2019 (박스권+상승 혼재)", "2016-01-01", "2018-12-31"),
    ("2019-2022 (코로나 폭락+급반등+2022 하락장)", "2019-01-01", "2021-12-31"),
    ("2022-2025 (금리인상+반도체 랠리)", "2022-01-01", "2024-12-31"),
    ("2016-2025 (전체)", "2016-01-01", None),  # None -> 오늘까지
]


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    today = pd.Timestamp.today().date().isoformat()

    print(f"=== v2 포트폴리오 전략: 서로 다른 국면 {len(WINDOWS)}개 구간 검증 ===")
    print(f"유니버스 상위 {top_n}개, 보유 {top_k}종목, 데이터는 {FETCH_START}부터 한 번만 조회\n")

    universe = get_universe(markets=("KOSPI", "KOSDAQ"), top_n=top_n)
    print(f"{len(universe)}개 종목 데이터 조회 중... (병렬)")

    def _fetch(code: str):
        return code, fetch_ohlcv_range(code, FETCH_START, today, warmup_days=WARMUP_DAYS)

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
        "KOSPI": fetch_ohlcv_range("KS11", FETCH_START, today, warmup_days=WARMUP_DAYS),
        "KOSDAQ": fetch_ohlcv_range("KQ11", FETCH_START, today, warmup_days=WARMUP_DAYS),
    }
    print(f"{len(price_data)}개 종목 확보 완료.\n")

    rows = []
    for label, start, end in WINDOWS:
        end = end or today
        try:
            result = run_portfolio_backtest(
                price_data, market_by_code, index_data,
                start=pd.Timestamp(start), end=pd.Timestamp(end), top_k=top_k,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{label}] 실패: {e}")
            continue

        kospi_period = index_data["KOSPI"]
        kospi_period = kospi_period[
            (kospi_period.index >= pd.Timestamp(start)) & (kospi_period.index <= pd.Timestamp(end))
        ]
        kospi_bh = buy_and_hold_return_pct(kospi_period) if not kospi_period.empty else None

        rows.append({
            "구간": label,
            "기간": f"{result.equity_curve.index.min().date()} ~ {result.equity_curve.index.max().date()}",
            "누적수익률": result.total_return_pct,
            "CAGR": result.cagr,
            "MDD": result.mdd_pct,
            "샤프": result.sharpe_ratio,
            "매매횟수": result.num_trades,
            "ATR손절": result.num_stopped_out,
            "코스피단순보유": kospi_bh,
        })

    print(f"{'구간':<45} {'누적수익률':>10} {'CAGR':>8} {'MDD':>8} {'샤프':>6} "
          f"{'매매':>5} {'ATR손절':>7} {'코스피보유':>10}")
    for r in rows:
        cagr_s = f"{r['CAGR']:+.1%}" if r["CAGR"] is not None else "—"
        sharpe_s = f"{r['샤프']:.2f}" if r["샤프"] is not None else "—"
        bh_s = f"{r['코스피단순보유']:+.1%}" if r["코스피단순보유"] is not None else "—"
        print(
            f"{r['구간']:<45} {r['누적수익률']:>+9.1%} {cagr_s:>8} {r['MDD']:>+7.1%} "
            f"{sharpe_s:>6} {r['매매횟수']:>5}회 {r['ATR손절']:>6}회 {bh_s:>10}"
        )


if __name__ == "__main__":
    main()
