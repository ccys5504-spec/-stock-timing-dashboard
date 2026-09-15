"""워크포워드(walk-forward) 검증 — "과거 전체에 최적화한 파라미터가 그 과거에서
잘 맞는 건 당연하다"는 과최적화(overfitting) 문제를 줄이기 위한 검증 방식.

scripts/optimize.py는 상승장 구간과 2022년 하락장 구간 "전체"를 보고 나서
그 구간에서 가장 잘 맞았던 조합을 고른다. 이건 시험 문제와 답을 같이 준
다음 "이 학생 몇 점이나 받나 보자"라고 채점하는 것과 같아서, 그 조합이
"다음에도" 잘 맞을지는 사실 이 결과만으로는 알 수 없다.

워크포워드 검증은 이렇게 한다:
  1) 과거를 "훈련 구간"과, 그 바로 다음 "검증 구간"으로 나눈다.
  2) 파라미터 조합 탐색(격자 탐색)은 훈련 구간만 보고 한다 — 검증 구간은
     탐색 중에는 전혀 들여다보지 않는다("눈가림").
  3) 훈련 구간에서 고른 조합을, 훈련 때는 보지 못했던 검증 구간에 그대로
     적용해서 성과를 잰다. 이게 "실전에 가까운" 성과 추정치다.
  4) 훈련/검증 구간을 시간축을 따라 한 칸씩 밀면서(rolling) 여러 번
     반복하고, 검증 구간 성과들을 평균 내서 "운 좋은 한 구간" 효과를 줄인다.

이 검증 구간(out-of-sample) 성과가 scripts/optimize.py가 보여주는 "전체
구간 최적화" 성과보다 낮게 나오는 건 정상이고 오히려 건강한 신호다 —
두 수치의 차이가 클수록 과최적화가 심했다는 뜻이다.

사용법: python scripts/walk_forward.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _pyarrow_compat  # noqa: F401,E402  # pandas보다 반드시 먼저 임포트

import pandas as pd  # noqa: E402

from data.fetch import WATCHLIST, fetch_ohlcv_range  # noqa: E402
from signals.indicators import add_indicators, add_scores  # noqa: E402
from backtest.engine import run_backtest, buy_and_hold_return_pct  # noqa: E402

TRAIN_YEARS = 3
TEST_YEARS = 1
# 처음 훈련 구간이 시작하는 지점. 여기서부터 TEST_YEARS(1년)씩 밀어가며
# "훈련 3년 -> 검증 1년" 폴드를 오늘까지 최대한 만든다.
FIRST_TRAIN_START = dt.date(2016, 1, 1)

# 훈련 구간에서 탐색할 조합 (scripts/optimize.py보다 축소 — 폴드 수가 많아서
# 전부 곱하면 시간이 오래 걸린다. 거래량 필터는 제외하고 핵심 파라미터만 본다).
MA_COMBOS = [(10, 40), (20, 60), (20, 90), (30, 90), (10, 60)]
RSI_COMBOS = [(25, 75), (30, 70), (35, 65)]
THRESHOLDS = [1, 2]
STOP_LOSSES = [None, 0.10, 0.15]

# 훈련 구간(3년) 동안 이 정도는 실제로 매매해야 "전략"으로 인정한다 — 매매를
# 거의 안 해서 우연히 손실을 피한 조합이 뽑히는 걸 막기 위함(scripts/optimize.py
# 와 동일한 취지).
MIN_TRAIN_TRADES = 3


def _make_folds(today: dt.date) -> list[tuple[dt.date, dt.date, dt.date]]:
    """(훈련 시작일, 검증 시작일, 검증 종료일) 튜플 리스트를 만든다."""
    folds = []
    train_start = FIRST_TRAIN_START
    while True:
        test_start = dt.date(train_start.year + TRAIN_YEARS, train_start.month, train_start.day)
        test_end = dt.date(test_start.year + TEST_YEARS, test_start.month, test_start.day) - dt.timedelta(days=1)
        if test_end > today:
            break
        folds.append((train_start, test_start, test_end))
        train_start = dt.date(train_start.year + TEST_YEARS, train_start.month, train_start.day)
    return folds


def _best_combo_on_train(df_ind_by_ma: dict, train_start: pd.Timestamp, train_end: pd.Timestamp):
    """훈련 구간만 보고 가장 성과 좋은 (거래 충분한) 조합을 고른다."""
    best = None
    for (ma_short, ma_long), df_ind in df_ind_by_ma.items():
        for rsi_os, rsi_ob in RSI_COMBOS:
            for threshold in THRESHOLDS:
                for stop_loss in STOP_LOSSES:
                    df_scored = add_scores(
                        df_ind, buy_threshold=threshold, sell_threshold=-threshold,
                        rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
                    )
                    train_slice = df_scored[(df_scored.index >= train_start) & (df_scored.index < train_end)]
                    if train_slice.empty:
                        continue
                    r = run_backtest(train_slice, stop_loss_pct=stop_loss)
                    if r.num_trades < MIN_TRAIN_TRADES:
                        continue
                    candidate = {
                        "ma": (ma_short, ma_long), "rsi": (rsi_os, rsi_ob),
                        "threshold": threshold, "stop_loss": stop_loss,
                        "train_return": r.total_return_pct,
                    }
                    if best is None or candidate["train_return"] > best["train_return"]:
                        best = candidate
    return best


def run_stock(code: str, today: dt.date) -> list[dict]:
    folds = _make_folds(today)
    rows = []
    for train_start, test_start, test_end in folds:
        try:
            raw = fetch_ohlcv_range(code, train_start.isoformat(), test_end.isoformat(), warmup_days=300)
        except Exception:  # noqa: BLE001
            continue
        if raw.index.min().date() > train_start + dt.timedelta(days=60):
            # 상장일이 훈련 시작일보다 한참 뒤라 이 종목은 이 폴드에서 아직
            # 존재하지 않았다 — 건너뛴다(있지도 않았던 종목을 백테스트할 수 없음).
            continue

        df_ind_by_ma = {}
        for ma_short, ma_long in MA_COMBOS:
            df_ind_by_ma[(ma_short, ma_long)] = add_indicators(
                raw, ma_short=ma_short, ma_long=ma_long
            ).dropna(subset=["MA_LONG"])

        train_ts = pd.Timestamp(train_start)
        test_start_ts = pd.Timestamp(test_start)
        test_end_ts = pd.Timestamp(test_end)

        best = _best_combo_on_train(df_ind_by_ma, train_ts, test_start_ts)
        if best is None:
            continue

        # 검증(out-of-sample) 구간에 훈련 때 고른 조합을 "그대로" 적용
        ma_short, ma_long = best["ma"]
        rsi_os, rsi_ob = best["rsi"]
        df_ind_test = df_ind_by_ma[(ma_short, ma_long)]
        df_scored_test = add_scores(
            df_ind_test, buy_threshold=best["threshold"], sell_threshold=-best["threshold"],
            rsi_oversold=rsi_os, rsi_overbought=rsi_ob,
        )
        test_slice = df_scored_test[
            (df_scored_test.index >= test_start_ts) & (df_scored_test.index <= test_end_ts)
        ]
        if test_slice.empty:
            continue

        r_test = run_backtest(test_slice, stop_loss_pct=best["stop_loss"])
        bh_test = buy_and_hold_return_pct(test_slice)

        sl_label = "없음" if best["stop_loss"] is None else f"{best['stop_loss']:.0%}"
        rows.append({
            "train": f"{train_start}~{test_start}", "test": f"{test_start}~{test_end}",
            "combo": f"MA{ma_short}/{ma_long} RSI{rsi_os}/{rsi_ob} ±{best['threshold']} 손절{sl_label}",
            "train_return": best["train_return"],
            "test_return": r_test.total_return_pct,
            "test_bh_return": bh_test,
            "test_trades": r_test.num_trades,
        })
    return rows


def main() -> None:
    today = dt.date.today()
    print(f"=== 워크포워드 검증 (훈련 {TRAIN_YEARS}년 -> 검증 {TEST_YEARS}년, 오늘 {today} 기준) ===\n")

    all_test_returns = []
    all_bh_returns = []

    for name, code in WATCHLIST.items():
        print(f"\n{'=' * 78}\n[{name} ({code})]\n{'=' * 78}")
        rows = run_stock(code, today)
        if not rows:
            print("  ⚠️ 충분한 데이터 폴드를 만들지 못했습니다 (상장일이 늦거나 데이터 부족).\n")
            continue

        print(f"  {'훈련구간':>21} {'검증구간':>21} | {'훈련수익률':>9} {'검증(OOS)수익률':>14} "
              f"{'검증기간 단순보유':>16} {'검증거래':>7}")
        for r in rows:
            print(
                f"  {r['train']:>21} {r['test']:>21} | {r['train_return']:>+8.1%} "
                f"{r['test_return']:>+13.1%} {r['test_bh_return']:>+15.1%} {r['test_trades']:>6}회"
            )
            print(f"    └ 훈련에서 고른 조합: {r['combo']}")
            all_test_returns.append(r["test_return"])
            all_bh_returns.append(r["test_bh_return"])

        avg_test = sum(r["test_return"] for r in rows) / len(rows)
        avg_bh = sum(r["test_bh_return"] for r in rows) / len(rows)
        win_folds = sum(1 for r in rows if r["test_return"] > r["test_bh_return"])
        print(
            f"\n  → 이 종목의 검증(OOS) 구간 평균 수익률: {avg_test:+.1%} "
            f"(같은 구간 단순보유 평균 {avg_bh:+.1%}) · 전략이 단순보유를 이긴 폴드: "
            f"{win_folds}/{len(rows)}"
        )

    if all_test_returns:
        overall_test = sum(all_test_returns) / len(all_test_returns)
        overall_bh = sum(all_bh_returns) / len(all_bh_returns)
        overall_wins = sum(1 for t, b in zip(all_test_returns, all_bh_returns) if t > b)
        print(f"\n{'=' * 78}")
        print(
            f"전체 종목·전체 폴드 평균 — 검증(OOS) 수익률: {overall_test:+.1%} / "
            f"단순보유: {overall_bh:+.1%} / 전략이 이긴 폴드: {overall_wins}/{len(all_test_returns)}"
        )
        print(
            "⚠️ 이 수치를 scripts/optimize.py의 '전체 구간 최적화' 수치와 비교해보면, "
            "워크포워드(검증 구간을 못 보고 고른) 쪽이 대체로 더 낮게 나온다 — "
            "그 차이가 지금까지 README에 있던 최적화 수치에 섞여 있던 과최적화 효과의 "
            "대략적인 크기다."
        )


if __name__ == "__main__":
    main()
