"""v2 구성요소를 하나씩 끄거나 바꿔서 "수익률에 기여하는가, 깎아먹는가"를 잰다.

배경(2026-09-20): v2는 같은 종목군을 동일가중으로 들고만 있었을 때(+676.9%)의 절반
남짓(+356.7%)만 벌었다. 수익을 높이려면 어떤 구성요소가 수익을 깎는지 알아야 한다.
파라미터를 그리드로 훑으면 v1의 optimize.py처럼 과최적화되므로, 여기서는
  - 연속값을 튜닝하지 않고 "켬/끔" 또는 "표준값 vs 2배" 같은 소수의 이산 변형만 쓰고
  - 전체 기간 하나가 아니라 서로 다른 국면 구간마다 같은 방향으로 개선되는지 본다.

⚠️ 결과를 보고 최고 수익 조합을 고르는 순간 이것도 과최적화다. 채택 기준은
"모든 구간에서 개선되거나, 최소한 어느 구간에서도 크게 악화되지 않음"이어야 한다.

사용법: python scripts/portfolio_ablation.py [유니버스크기] [보유종목수] [core]
  마지막에 core를 붙이면 '기준'과 '장세필터 끔' 두 변형만 돌린다(민감도 재확인용).
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

from _common import load_universe_prices, report_load  # noqa: E402
from backtest.benchmarks import equal_weight_equity, summarize_equity  # noqa: E402
from backtest.portfolio_engine import (  # noqa: E402
    ATR_INITIAL_MULT, ATR_TRAIL_MULT, run_portfolio_backtest,
)
from data.fetch import fetch_ohlcv_range, get_universe  # noqa: E402

FETCH_START = "2015-06-01"
WARMUP_DAYS = 450

WINDOWS = [
    ("2016-18", "2016-01-01", "2018-12-31"),
    ("2019-21", "2019-01-01", "2021-12-31"),
    ("2022-24", "2022-01-01", "2024-12-31"),
    ("2025-오늘", "2025-01-01", None),
    ("전체", "2016-01-01", None),
]

VARIANTS: list[tuple[str, dict]] = [
    ("기준(현재 v2)", {}),
    ("장세필터 끔", dict(use_regime=False)),
    ("ATR손절 끔", dict(atr_initial_mult=None, atr_trail_mult=None)),
    ("ATR손절 2배 넓게", dict(atr_initial_mult=ATR_INITIAL_MULT * 2, atr_trail_mult=ATR_TRAIL_MULT * 2)),
    ("동일가중(역변동성 끔)", dict(weighting="equal")),
    ("장세 끔 + 손절 2배", dict(use_regime=False, atr_initial_mult=ATR_INITIAL_MULT * 2,
                              atr_trail_mult=ATR_TRAIL_MULT * 2)),
    ("장세 끔 + 손절 끔", dict(use_regime=False, atr_initial_mult=None, atr_trail_mult=None)),
    ("전부 끔(순수 모멘텀)", dict(use_regime=False, atr_initial_mult=None, atr_trail_mult=None,
                             weighting="equal")),
]


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    core_only = len(sys.argv) > 3 and sys.argv[3] == "core"
    variants = VARIANTS[:2] if core_only else VARIANTS
    today = pd.Timestamp.today().date().isoformat()

    print(f"=== v2 구성요소 기여도 실험 (유니버스 상위 {top_n}개, 보유 {top_k}종목, 왕복비용 0.3%) ===\n")
    universe = get_universe(markets=("KOSPI", "KOSDAQ"), top_n=top_n)
    price_data, market_by_code, failed = load_universe_prices(universe, FETCH_START, today)
    report_load(len(universe), price_data, failed)
    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", FETCH_START, today, warmup_days=WARMUP_DAYS),
        "KOSDAQ": fetch_ohlcv_range("KQ11", FETCH_START, today, warmup_days=WARMUP_DAYS),
    }

    print(f"\n{'변형':<22}" + "".join(f"{w[0]:>22}" for w in WINDOWS))
    print(f"{'':<22}" + "".join(f"{'수익률 / MDD':>22}" for _ in WINDOWS))
    for name, kwargs in variants:
        cells = []
        for _label, start, end in WINDOWS:
            r = run_portfolio_backtest(
                price_data, market_by_code, index_data,
                start=pd.Timestamp(start), end=pd.Timestamp(end or today), top_k=top_k, **kwargs,
            )
            cells.append(f"{r.total_return_pct:+8.1%} / {r.mdd_pct:+6.1%}")
        print(f"{name:<22}" + "".join(f"{c:>22}" for c in cells))

    cells = []
    for _label, start, end in WINDOWS:
        total, mdd = summarize_equity(
            equal_weight_equity(price_data, pd.Timestamp(start), pd.Timestamp(end or today))
        )
        cells.append(f"{total:+8.1%} / {mdd:+6.1%}")
    print(f"{'[참고] 동일가중 보유':<22}" + "".join(f"{c:>22}" for c in cells))
    print("\n※ 동일가중 보유는 생존편향이 섞인 상한선이며 비용 미반영입니다.")


if __name__ == "__main__":
    main()
