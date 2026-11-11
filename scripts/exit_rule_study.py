"""사용자가 정한 매도 규칙(손절 -3%, +5/+10/+20% 이익 보호선)이 과거에 어떻게 작동했는지 본다.

방법: 2016~2026 매월 첫 거래일 시가에, 그 시점 시가총액 상위 N개 종목(상장폐지 포함 — data/pit_cache)을 각각
샀다고 가정하고, (a) 규칙대로 팔 때와 (b) 규칙 없이 HORIZON 거래일 뒤 종가에 팔 때를 비교한다. 규칙으로 안 팔린
경우도 HORIZON 거래일 뒤 종가에 판다. 왕복 비용 0.3%를 양쪽에 똑같이 뺀다.

⚠️ 이건 '어떤 종목을 살지'가 아니라 '산 다음 규칙이 어떤 효과를 내는지'만 본다. 같은 종목·같은 날짜에 사서 규칙 유무만
다르므로 비교는 공정하지만, 일봉으로 체결을 가정한 것이라 갭하락/하한가의 실제 체결은 더 나쁠 수 있다.

전제: scripts/build_pit_dataset.py로 data/pit_cache/를 만들어 둘 것.
사용법: python scripts/exit_rule_study.py [종목수=30] [보유 거래일=60]
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _pyarrow_compat  # noqa: F401,E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from portfolio_pit_backtest import load_cache  # noqa: E402
from strategy.exit_rules import DEFAULT_RULE, ExitRule, simulate_exit  # noqa: E402
from strategy.momentum import rank_universe  # noqa: E402

COST = 0.003
START, END = "2016-01-01", None


def month_first_days(calendar: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(calendar, index=calendar)
    return s.groupby(calendar.to_period("M")).min().tolist()


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    horizon = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    price_data, meta = load_cache()
    shares = dict(zip(meta["Code"], meta["Shares"]))
    kospi = price_data["005930"].index  # 달력: 삼성전자가 거래된 날
    calendar = kospi[kospi >= pd.Timestamp(START)]
    entries = month_first_days(calendar)[:-3]  # 마지막 석 달은 보유 기간이 모자라 제외

    samples = []  # (진입일, 코드, 진입 이후 일봉 window, 진입가)
    for entry_day in entries:
        prev_close_day = kospi[kospi.get_loc(entry_day) - 1]
        ranked = rank_universe(price_data, prev_close_day, top_n=top_n, shares=shares)
        for code in ranked["코드"]:  # 시가총액 상위 top_n개(정렬은 모멘텀이지만 여기서는 멤버십만 쓴다)
            df = price_data[code]
            if entry_day not in df.index:
                continue
            pos = df.index.get_loc(entry_day)
            window = df.iloc[pos: pos + horizon]
            entry_price = float(window["Open"].iloc[0])
            if len(window) < 2 or entry_price <= 0:
                continue  # 중간에 상장폐지된 경우는 그때까지의 봉으로 계산한다
            samples.append((entry_day, code, window, entry_price))

    def run(rule: ExitRule) -> pd.DataFrame:
        rows = []
        for entry_day, code, window, entry_price in samples:
            res = simulate_exit(window, entry_price, rule)
            hold_ret = float(window["Close"].iloc[-1]) / entry_price - 1
            if res.triggered:
                rule_ret, days, kind = res.exit_price / entry_price - 1, res.exit_index + 1, rule.label(res.exit_floor)
            else:
                rule_ret, days, kind = hold_ret, len(window), "기간 만료"
            rows.append({"date": entry_day, "rule": rule_ret - COST, "hold": hold_ret - COST, "days": days,
                         "kind": kind, "early_stop": bool(res.triggered and res.exit_floor < 0 and days <= 5)})
        d = pd.DataFrame(rows)
        d["period"] = np.where(d["date"] < pd.Timestamp("2021-01-01"), "2016-2020", "2021-")
        return d

    def summarize(d: pd.DataFrame, name: str) -> None:
        print(f"[{name}] n={len(d):,}")
        print(f"  평균수익  규칙 {d['rule'].mean():+.2%} | 규칙 없이 보유 {d['hold'].mean():+.2%}")
        print(f"  중앙값    규칙 {d['rule'].median():+.2%} | 규칙 없이 보유 {d['hold'].median():+.2%}")
        print(f"  승률      규칙 {(d['rule'] > 0).mean():.1%} | 규칙 없이 보유 {(d['hold'] > 0).mean():.1%}")
        print(f"  최악 5%   규칙 {d['rule'].quantile(0.05):+.2%} | 규칙 없이 보유 {d['hold'].quantile(0.05):+.2%}")
        print(f"  최악 1건  규칙 {d['rule'].min():+.2%} | 규칙 없이 보유 {d['hold'].min():+.2%}")
        print(f"  평균 보유일 {d['days'].mean():.1f}일(규칙 없이 보유는 최대 {horizon}일), 5일 안에 손절된 비율 {d['early_stop'].mean():.1%}")
        print("  매도 사유: " + ", ".join(f"{k} {v:.1%}" for k, v in d["kind"].value_counts(normalize=True).items()))
        print()

    print(f"=== 매도 규칙 효과 연구: 매월 첫날 시총 상위 {top_n}개 매수, 최대 {horizon}거래일 보유, 비용 0.3% ===")
    print(f"표본 {len(samples):,}건 (월 {len(entries)}회 x 종목)\n")
    base = run(DEFAULT_RULE)
    print("##### 사용자 규칙: 손절 -3%, 이익 보호선 +5/+10/+20% #####\n")
    summarize(base, "전체")
    for p, g in base.groupby("period"):
        summarize(g, p)
    stopped = base[base["kind"] == "손절 -3%"]
    if len(stopped):
        print(f"손절 -3%로 팔린 {len(stopped):,}건 중 규칙 없이 끝까지 들고 있었다면 이익이었던 비율: "
              f"{(stopped['hold'] > 0).mean():.1%} (평균 {stopped['hold'].mean():+.2%})\n")

    print("##### 손절폭만 바꿔 본 비교 (이익 보호선 +5/+10/+20%는 그대로) #####")
    print(f"{'손절폭':>6} {'평균수익':>9} {'중앙값':>8} {'승률':>7} {'최악5%':>8} {'평균보유일':>9} {'손절비율':>8}")
    for stop in (0.03, 0.05, 0.07, 0.10, 0.15):
        d = run(ExitRule(stop_loss=stop))
        stop_share = d["kind"].str.startswith("손절").mean()
        print(f"{-stop:>6.0%} {d['rule'].mean():>+9.2%} {d['rule'].median():>+8.2%} {(d['rule'] > 0).mean():>7.1%} "
              f"{d['rule'].quantile(0.05):>+8.2%} {d['days'].mean():>8.1f}일 {stop_share:>8.1%}")
    d = run(ExitRule(stop_loss=0.03))
    print(f"{'규칙없이':>6} {d['hold'].mean():>+9.2%} {d['hold'].median():>+8.2%} {(d['hold'] > 0).mean():>7.1%} "
          f"{d['hold'].quantile(0.05):>+8.2%} {float(horizon):>8.1f}일 {'—':>8}")


if __name__ == "__main__":
    main()
