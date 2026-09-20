"""종목 선택(모멘텀 순위)이 실제로 값어치를 더하는지 대조군과 비교한다.

편향 없는 종목군(시점 기준 시총 상위 100개, 상장폐지 포함)에서 v2 전체 성과(+93.8%)가 같은
종목군 동일가중(+75.8%)과 비슷했다. "모멘텀으로 고른 N종목"이 "같은 후보군에서 아무렇게나 고른
N종목"보다 나은지 본다. 나머지 규칙(비중·순위이력·장세·비용)은 전부 동일하고 후보 정렬 기준만 바꾼다:
  momentum  — 현재 v2
  random    — 무작위(시드 N개의 분포로 비교)
  large_cap — 시가총액 큰 순
  low_vol   — 최근 60일 변동성 낮은 순

사용법: python scripts/portfolio_pit_selection_test.py [무작위 시드 수=30] [보유종목수=엔진 기본값]
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pyarrow_compat  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from portfolio_pit_backtest import FETCH_START, load_cache  # noqa: E402
from backtest.portfolio_engine import TOP_K, run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range  # noqa: E402

WINDOWS = [
    ("2016-2020", "2016-01-01", "2020-12-31"),
    ("2021-오늘", "2021-01-01", None),
    ("전체", "2016-01-01", None),
]


def main() -> None:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    top_k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K
    today = pd.Timestamp.today().normalize()
    price_data, meta = load_cache()
    market_by_code = dict(zip(meta["Code"], meta["Market"]))
    shares = dict(zip(meta["Code"], meta["Shares"]))
    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", FETCH_START, today.date().isoformat(), warmup_days=450),
        "KOSDAQ": fetch_ohlcv_range("KQ11", FETCH_START, today.date().isoformat(), warmup_days=450),
    }

    def run(start, end, **kw):
        return run_portfolio_backtest(
            price_data, market_by_code, index_data, pd.Timestamp(start), pd.Timestamp(end) if end else today,
            top_k=top_k, universe_top_n=100, shares=shares, **kw,
        )

    print("=== 종목 선택 대조 실험 (시점 기준 시총 상위 100개, 15종목, 장세 켬·손절 끔, 왕복비용 0.3%) ===\n")
    print(f"{'구성':<22}" + "".join(f"{w[0]:>26}" for w in WINDOWS))
    print(f"{'':<22}" + "".join(f"{'수익률 / MDD':>26}" for _ in WINDOWS))
    for label, kw in [("모멘텀 (현재 v2)", dict(ranking="momentum")), ("시가총액 큰 순", dict(ranking="large_cap")),
                      ("저변동성 순", dict(ranking="low_vol"))]:
        cells = []
        for _n, start, end in WINDOWS:
            r = run(start, end, **kw)
            cells.append(f"{r.total_return_pct:+8.1%} / {r.mdd_pct:+6.1%}")
        print(f"{label:<22}" + "".join(f"{c:>26}" for c in cells), flush=True)

    rand = {w[0]: [] for w in WINDOWS}
    for seed in range(n_seeds):
        for name, start, end in WINDOWS:
            rand[name].append(run(start, end, ranking="random", ranking_seed=seed).total_return_pct)
        print(f"  (무작위 {seed + 1}/{n_seeds})", flush=True) if (seed + 1) % 10 == 0 else None

    mom = {name: run(start, end, ranking="momentum").total_return_pct for name, start, end in WINDOWS}
    print(f"\n무작위 {top_k}종목 {n_seeds}회 분포 (같은 후보군·같은 규칙):")
    print(f"{'구간':<12}{'평균':>10}{'중앙값':>10}{'5%':>10}{'95%':>10}{'모멘텀':>10}{'모멘텀보다 나은 무작위':>26}")
    for name, _s, _e in WINDOWS:
        v = np.array(rand[name])
        better = int((v >= mom[name]).sum())
        print(f"{name:<12}{v.mean():>+10.1%}{np.median(v):>+10.1%}{np.percentile(v, 5):>+10.1%}"
              f"{np.percentile(v, 95):>+10.1%}{mom[name]:>+10.1%}{f'{better}/{n_seeds}회':>26}")
    print("\n※ '모멘텀보다 나은 무작위'가 절반 안팎이면 모멘텀 순위가 무작위 선택과 구별되지 않는다는 뜻입니다.")


if __name__ == "__main__":
    main()
