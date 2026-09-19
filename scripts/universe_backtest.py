"""관심 종목 3개가 아니라, 더 넓은 종목군에 "앱 기본 설정 그대로"의 전략을
적용했을 때 결과가 어떻게 퍼지는지 본다.

이걸 만든 이유 (README/외부 코드 리뷰에서 지적된 두 가지 문제):

1) **표본이 3개뿐이라 일반화하기 어려움** — SK하이닉스/두산에너빌리티/
   주성엔지니어링은 원래 사용자가 "관심 있어서" 고른 종목이지, 무작위로 뽑힌
   종목이 아니다. 이 3개에서 전략이 잘 맞았다고 해서 다른 종목에도 잘
   맞으리란 보장은 없다. 이 스크립트는 시가총액 상위 종목들에 **똑같은
   기본 설정**(MA20/60, RSI 30/70, 임계값 ±1, 거래량 필터 자동, 손절 없음 —
   scripts/optimize.py처럼 종목마다 파라미터를 따로 맞추지 않음)을 그대로
   적용해서, "평균적으로" 도움이 되는지를 본다.

2) **생존편향(survivorship bias)** — 아래 종목 목록은 **오늘 시점** 시가총액
   상위 종목이다. 3~5년 전에는 목록이 달랐을 것이고, 그 사이 상장폐지되거나
   크게 망한 종목은 오늘 목록에 아예 안 잡히므로 이 백테스트에는 등장하지
   않는다. 즉 "살아남은 종목들"만 보고 있는 셈이라 결과가 실제보다 낙관적으로
   나올 가능성이 있다 — FinanceDataReader로는 과거 시점의 상장폐지 종목
   목록을 구하기 어려워서 이 한계를 완전히 없애지는 못했다. 이 스크립트의
   결과를 볼 때는 이 점을 감안해야 한다.

사용법: python scripts/universe_backtest.py [top_n]
"""
from __future__ import annotations

import statistics
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Windows 콘솔(cp949)에서 ⚠️ 같은 이모지를 출력하면 UnicodeEncodeError가 나는
# 경우가 있어, 표준출력을 UTF-8로 강제한다 (PowerShell/cmd에서 직접 실행해도
# 안전하도록).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _pyarrow_compat  # noqa: F401,E402  # pandas보다 반드시 먼저 임포트

import pandas as pd  # noqa: E402

from data.fetch import get_universe, fetch_ohlcv, fetch_ohlcv_range  # noqa: E402
from signals.indicators import build_signals  # noqa: E402
from backtest.engine import run_backtest, buy_and_hold_return_pct  # noqa: E402

MAX_WORKERS = 8
BEAR_START, BEAR_END = "2022-01-01", "2022-12-31"
# 앱의 기본 설정 그대로 (signals/indicators.py 모듈 기본값과 동일) — 종목별로
# 파라미터를 따로 맞추지 않는다. 그게 이 스크립트의 요점이다.
DEFAULT_KWARGS = dict(buy_threshold=1, sell_threshold=-1, volume_filter="auto")


def _one_stock(code: str, name: str, years: int):
    try:
        df = fetch_ohlcv(code, years=years)
        df_sig = build_signals(df, **DEFAULT_KWARGS).dropna(subset=["MA_LONG"])
        cutoff = df_sig.index.max() - pd.Timedelta(days=int(years * 365.25))
        df_period = df_sig[df_sig.index >= cutoff]
        if df_period.empty:
            return None
        result = run_backtest(df_period)
        bh = buy_and_hold_return_pct(df_period)
        return {
            "종목명": name, "종목코드": code,
            "전략수익률": result.total_return_pct, "단순보유수익률": bh,
            "초과수익": result.total_return_pct - bh,
            "거래수": result.num_trades, "MDD": result.mdd_pct,
        }
    except Exception:  # noqa: BLE001
        return None


def _one_stock_bear(code: str, name: str):
    try:
        df = fetch_ohlcv_range(code, BEAR_START, BEAR_END)
        df_sig = build_signals(df, **DEFAULT_KWARGS).dropna(subset=["MA_LONG"])
        df_period = df_sig[(df_sig.index >= pd.Timestamp(BEAR_START)) & (df_sig.index <= pd.Timestamp(BEAR_END))]
        if df_period.empty:
            return None
        result = run_backtest(df_period)
        bh = buy_and_hold_return_pct(df_period)
        return {
            "종목명": name, "종목코드": code,
            "전략수익률": result.total_return_pct, "단순보유수익률": bh,
            "초과수익": result.total_return_pct - bh,
        }
    except Exception:  # noqa: BLE001
        return None


def _run_parallel(candidates: pd.DataFrame, fn) -> list[dict]:
    """종목별 결과를 모은다. 결과를 못 낸 종목(조회 실패/구간 데이터 없음)은
    조용히 빠지지 않도록 개수와 이름을 출력한다(2026-09-20 점검서 지적)."""
    rows = []
    missing = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fn, row["Code"], row["Name"]): row["Name"] for _, row in candidates.iterrows()
        }
        for future in as_completed(futures):
            r = future.result()
            if r is not None:
                rows.append(r)
            else:
                missing.append(futures[future])
    if missing:
        print(f"  ⚠️ 요청 {len(candidates)}개 중 {len(missing)}개는 결과가 없어 제외됨: {', '.join(sorted(missing))}")
    return rows


def _summarize(rows: list[dict], label: str) -> None:
    if not rows:
        print(f"  [{label}] 결과 없음\n")
        return
    strat = [r["전략수익률"] for r in rows]
    bh = [r["단순보유수익률"] for r in rows]
    excess = [r["초과수익"] for r in rows]
    beat = sum(1 for e in excess if e > 0)
    print(f"  [{label}] — {len(rows)}개 종목")
    print(f"    전략 수익률   : 평균 {statistics.mean(strat):+.1%} / 중앙값 {statistics.median(strat):+.1%}")
    print(f"    단순보유 수익률: 평균 {statistics.mean(bh):+.1%} / 중앙값 {statistics.median(bh):+.1%}")
    print(f"    전략이 단순보유를 이긴 종목: {beat}/{len(rows)}개 ({beat / len(rows):.0%})")
    print(f"    초과수익(전략-보유) 평균 {statistics.mean(excess):+.1%} / 중앙값 {statistics.median(excess):+.1%}")
    best = max(rows, key=lambda r: r["초과수익"])
    worst = min(rows, key=lambda r: r["초과수익"])
    print(f"    가장 도움 된 종목: {best['종목명']} (초과수익 {best['초과수익']:+.1%})")
    print(f"    가장 손해 본 종목: {worst['종목명']} (초과수익 {worst['초과수익']:+.1%})")
    print()


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(f"=== 시가총액 상위 {top_n}개 종목(KOSPI+KOSDAQ, 오늘 기준) 앱 기본 설정 백테스트 ===")
    print("⚠️ 아래 종목 목록은 '오늘' 시점 기준이라 생존편향이 있습니다 (스크립트 상단 설명 참고).\n")

    universe = get_universe(markets=("KOSPI", "KOSDAQ"), top_n=top_n)
    print(f"대상 종목 {len(universe)}개 조회 완료. 백테스트 중...\n")

    bull_rows = _run_parallel(universe, lambda c, n: _one_stock(c, n, years=3))
    _summarize(bull_rows, "최근 3년")

    bear_rows = _run_parallel(universe, lambda c, n: _one_stock_bear(c, n))
    _summarize(bear_rows, "2022년 하락장")

    print(
        "참고: 여기 쓰인 설정(MA20/60, RSI 30/70, 임계값 ±1, 거래량필터 자동, 손절 없음)은 "
        "종목별로 맞춘 게 아니라 앱을 처음 켰을 때의 기본값 그대로입니다. README의 다른 표들 "
        "(scripts/optimize.py, scripts/walk_forward.py 결과)은 SK하이닉스/두산에너빌리티/"
        "주성엔지니어링 3종목에 한해 파라미터를 맞춘 결과이니 직접 비교하지 마세요 — "
        "이 스크립트는 '더 넓은 종목에 손대지 않은 기본값을 적용하면 어떻게 되는지'를 보는 "
        "용도입니다."
    )


if __name__ == "__main__":
    main()
