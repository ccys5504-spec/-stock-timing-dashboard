"""생존편향을 줄인(시점 기준·상장폐지 포함) 종목군에서 v2 구성요소별 기여도를 다시 잰다.

scripts/portfolio_ablation.py는 "오늘의 시총 상위 100개"(생존편향 종목군)에서 잰 결과라, 거기서 나온
결론(예: 장세필터를 끄면 수익이 2배)이 편향 때문인지 알 수 없었다. 같은 실험을 편향을 줄인
종목군에서 다시 한다. 전제: scripts/build_pit_dataset.py로 data/pit_cache/를 만들어 둘 것.

사용법: python scripts/portfolio_pit_ablation.py [유니버스크기=100] [보유종목수=15]
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

from portfolio_pit_backtest import FETCH_START, WINDOWS, load_cache  # noqa: E402
from backtest.portfolio_engine import run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range  # noqa: E402

NO_STOP = dict(atr_initial_mult=None, atr_trail_mult=None)
VARIANTS: list[tuple[str, dict]] = [
    ("현재 기본(장세끔+ATR손절)", {}),
    ("장세필터 켬", dict(use_regime=True)),
    ("ATR손절 끔", NO_STOP),
    ("손절 끔 + 동일비중", dict(weighting="equal", **NO_STOP)),
    ("장세 켬 + 손절 끔", dict(use_regime=True, **NO_STOP)),
    ("종가기준 손절", dict(stop_on_close=True)),
]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top_n = int(args[0]) if len(args) > 0 else 100
    top_k = int(args[1]) if len(args) > 1 else 15
    today = pd.Timestamp.today().normalize()

    price_data, meta = load_cache()
    market_by_code = dict(zip(meta["Code"], meta["Market"]))
    shares = dict(zip(meta["Code"], meta["Shares"]))
    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", FETCH_START, today.date().isoformat(), warmup_days=450),
        "KOSDAQ": fetch_ohlcv_range("KQ11", FETCH_START, today.date().isoformat(), warmup_days=450),
    }
    print(f"=== 시점기준 종목군(상장폐지 포함) v2 구성요소 실험: 상위 {top_n}개 중 {top_k}종목, 왕복비용 0.3% ===\n")
    print(f"{'':<26}" + "".join(f"{w[0]:>22}" for w in WINDOWS))
    print(f"{'':<26}" + "".join(f"{'수익률 / MDD':>22}" for _ in WINDOWS))
    for name, kwargs in VARIANTS:
        cells = []
        for _label, start, end in WINDOWS:
            r = run_portfolio_backtest(
                price_data, market_by_code, index_data, pd.Timestamp(start),
                pd.Timestamp(end) if end else today, top_k=top_k, universe_top_n=top_n,
                shares=shares, **kwargs,
            )
            cells.append(f"{r.total_return_pct:+8.1%} / {r.mdd_pct:+6.1%}")
        print(f"{name:<26}" + "".join(f"{c:>22}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
