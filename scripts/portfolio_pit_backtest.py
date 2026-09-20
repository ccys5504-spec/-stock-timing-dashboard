"""생존편향을 줄인(상장폐지 종목 포함) v2 백테스트.

전제: `python scripts/build_pit_dataset.py`로 data/pit_cache/를 먼저 만들어 둘 것.

같은 규칙(매월 말 그날 시가총액(보정종가 x 상장주식수) 상위 N개를 후보로, 그 안에서 모멘텀 상위 K개)으로 비교한다:
  A. 시점 기준 전체 종목(상장폐지 포함)    <- 생존편향을 줄인 값
  B. 지금도 상장 중인 종목만(폐지 제외)      <- 예전 방식과 같은 종류의 편향이 있는 값
  A와 B의 차이 = 생존편향이 v2 수익률을 얼마나 부풀렸는지에 대한 추정
  기준선: 시점 기준 동일가중(상장폐지 포함), 코스피 단순보유

사용법: python scripts/portfolio_pit_backtest.py [유니버스크기=100] [보유종목수=엔진 기본값(3)] [--export 경로.json]
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pyarrow_compat  # noqa: F401,E402

import pandas as pd  # noqa: E402

from backtest.benchmarks import pit_equal_weight_equity, summarize_equity  # noqa: E402
from backtest.engine import buy_and_hold_return_pct  # noqa: E402
from backtest.portfolio_engine import TOP_K, run_portfolio_backtest  # noqa: E402
from data.fetch import fetch_ohlcv_range  # noqa: E402

CACHE_DIR = ROOT / "data" / "pit_cache"
FETCH_START = "2015-06-01"

WINDOWS = [
    ("2016-18", "2016-01-01", "2018-12-31"),
    ("2019-21", "2019-01-01", "2021-12-31"),
    ("2022-24", "2022-01-01", "2024-12-31"),
    ("2025-오늘", "2025-01-01", None),
    ("전체", "2016-01-01", None),
]


def load_cache() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    meta = pd.read_pickle(CACHE_DIR / "meta.pkl")
    price_data = {}
    for code in meta["Code"]:
        path = CACHE_DIR / f"{code}.pkl"
        if path.exists():
            with open(path, "rb") as f:
                df = pickle.load(f)
            if len(df):
                price_data[code] = df
    return price_data, meta


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    top_n = int(args[0]) if len(args) > 0 else 100
    top_k = int(args[1]) if len(args) > 1 else TOP_K
    export = sys.argv[sys.argv.index("--export") + 1] if "--export" in sys.argv else None
    today = pd.Timestamp.today().normalize()

    price_data, meta = load_cache()
    market_by_code = dict(zip(meta["Code"], meta["Market"]))
    shares = dict(zip(meta["Code"], meta["Shares"]))
    survivors = set(meta.loc[meta["DelistingDate"].isna(), "Code"]) & set(price_data)
    delisted = set(price_data) - survivors
    print(f"=== 생존편향 제거 v2 백테스트: 후보 = 매월 말 시가총액 근사 상위 {top_n}개, 보유 {top_k}종목 ===")
    print(f"시세 보유: 지금 상장 {len(survivors)}개 + 상장폐지 {len(delisted)}개 = {len(price_data)}개\n")

    index_data = {
        "KOSPI": fetch_ohlcv_range("KS11", FETCH_START, today.date().isoformat(), warmup_days=450),
        "KOSDAQ": fetch_ohlcv_range("KQ11", FETCH_START, today.date().isoformat(), warmup_days=450),
    }
    survivors_data = {c: price_data[c] for c in survivors}

    # (이름, 데이터, 인자, 모든 구간을 돌릴지) — 민감도 변형은 시간 절약을 위해 '전체' 구간만 돈다
    variants = [
        ("A 현재 기본(시총 큰 순·장세 켬·손절 끔)", price_data, {}, True),
        ("M 모멘텀 순위(비교용)", price_data, dict(ranking="momentum"), True),
        ("A2 폐지 종목 회수 -50% 가정", price_data, dict(delist_haircut=0.5), False),
        ("A3 왕복비용 1.0%", price_data, dict(round_trip_cost=0.010), False),
        ("B 현재 상장 종목만", survivors_data, {}, False),
    ]
    print(f"{'':<40}" + "".join(f"{w[0]:>22}" for w in WINDOWS))
    print(f"{'':<40}" + "".join(f"{'수익률 / MDD':>22}" for _ in WINDOWS))
    curves: dict[str, pd.Series] = {}
    stats: dict[str, dict] = {}
    for name, data, kwargs, all_windows in variants:
        cells = []
        for label, start, end in WINDOWS:
            if not all_windows and label != "전체":
                cells.append("—")
                continue
            r = run_portfolio_backtest(
                data, market_by_code, index_data, pd.Timestamp(start), pd.Timestamp(end) if end else today,
                top_k=top_k, universe_top_n=top_n, shares=shares, **kwargs,
            )
            cells.append(f"{r.total_return_pct:+8.1%} / {r.mdd_pct:+6.1%}")
            if label == "전체":
                curves[name] = r.equity_curve
                gone = [
                    t for t in r.trades
                    if not t.stopped_out and t.code in delisted
                    and price_data[t.code].index[-1] < today - pd.Timedelta(days=5)
                    and t.sell_date > price_data[t.code].index[-1]
                ]
                stats[name] = dict(
                    total=r.total_return_pct, cagr=r.cagr, mdd=r.mdd_pct, sharpe=r.sharpe_ratio,
                    trades=r.num_trades, stops=r.num_stopped_out, delisted_exits=len(gone),
                )
        print(f"{name:<40}" + "".join(f"{c:>22}" for c in cells), flush=True)

    for name, data in [("[기준선] 시점기준 동일가중", price_data), ("[기준선] 현재상장만 동일가중", survivors_data)]:
        cells = []
        for label, start, end in WINDOWS:
            cal = index_data["KOSPI"].index
            cal = cal[(cal >= pd.Timestamp(start)) & (cal <= (pd.Timestamp(end) if end else today))]
            eq = pit_equal_weight_equity(data, cal, top_n, shares=shares)
            total, mdd = summarize_equity(eq)
            cells.append(f"{total:+8.1%} / {mdd:+6.1%}")
            if label == "전체" and "시점기준" in name:
                curves["동일가중(시점기준)"] = eq
        print(f"{name:<40}" + "".join(f"{c:>22}" for c in cells), flush=True)
    cells = []
    for label, start, end in WINDOWS:
        kp = index_data["KOSPI"]
        kp = kp[(kp.index >= pd.Timestamp(start)) & (kp.index <= (pd.Timestamp(end) if end else today))]
        cells.append(f"{buy_and_hold_return_pct(kp):+8.1%}")
    print(f"{'[기준선] 코스피 보유':<40}" + "".join(f"{c:>22}" for c in cells))

    print("\n전체 기간 상세:")
    for name, st in stats.items():
        sh = f"{st['sharpe']:.2f}" if st["sharpe"] is not None else "-"
        print(f"  {name:<40} 수익 {st['total']:+.1%} CAGR {st['cagr']:+.1%} MDD {st['mdd']:+.1%} 샤프 {sh} "
              f"매매 {st['trades']}회 (손절 {st['stops']}회, 상장폐지로 청산 {st['delisted_exits']}회)")

    if export:
        kospi = index_data["KOSPI"]["Close"]
        kospi = kospi[(kospi.index >= pd.Timestamp("2016-01-01")) & (kospi.index <= today)]
        curves["코스피"] = kospi / kospi.iloc[0]

        def monthly(s: pd.Series) -> list[list]:
            m = s.groupby(s.index.to_period("M")).last()
            return [["2015-12", 1.0]] + [[str(k), round(float(v), 4)] for k, v in m.items()]  # 시작점(원금 1.0) 포함

        out = {
            "generated": today.date().isoformat(), "top_n": top_n, "top_k": top_k,
            "stats": stats, "curves": {k: monthly(v / v.iloc[0]) for k, v in curves.items()},
        }
        Path(export).parent.mkdir(parents=True, exist_ok=True)
        with open(export, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=1, default=float)
        print(f"\n결과를 {export}에 저장했습니다.")


if __name__ == "__main__":
    main()
