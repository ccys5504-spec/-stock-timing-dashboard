"""생존편향을 줄이기 위한 "그 시점에 실제로 거래되던 종목 전체" 데이터셋을 만든다.

지금까지 v2 백테스트는 **오늘 시점** 시가총액 상위 100개만 대상으로 했다. 그 목록에는
과거에 상장폐지되거나 크게 망해서 순위에서 밀려난 종목이 하나도 없어서, 결과가 실제보다
낙관적일 수밖에 없다(생존편향). 이 스크립트는 그 한계를 없애기 위해 두 곳의 무료 자료를 합친다:

1. 현재 상장 종목 전체: `fdr.StockListing("KRX")`
2. **상장폐지 종목 전체(1960~오늘, 4,000여 건, 폐지 사유·일자 포함)**: FinanceDataReader
   프로젝트가 KRX 통계에서 매일 모아 공개하는 캐시(FinanceData/fdr_krx_data_cache,
   data/listing/delisting/YYYY-MM-DD.csv). KRX 사이트를 직접 호출하는 방식은 지금 막혀 있어
   (FDR의 'KRX-DELISTING' 조회는 빈 결과) 이 캐시가 유일하게 동작하는 경로였다.
3. 상장폐지 종목의 **시세**: `fdr.DataReader(코드)`가 폐지된 종목도 폐지일 직전까지의
   일봉을 돌려준다(예: 한진해운 117930은 2017-03-06까지) — 폐지 직전 폭락 구간(정리매매)이
   그대로 들어 있다.

시세 주의: FDR(네이버) 일봉은 **가격만** 액면분할·병합에 맞춰 보정되고 거래량은 보정되지 않아,
"가격 x 거래량" 거래대금은 분할/병합 종목에서 크게 틀린다(삼성전자 2016년이 1/50로 계산됨).
그래서 시점 기준 유니버스는 거래대금이 아니라 **보정가격 x 상장주식수(시가총액 근사)**로 고른다
(현재 종목은 현재 주식수, 폐지 종목은 폐지 시점 주식수 — 증자/소각 이력은 반영 못 하는 근사).

결과는 `data/pit_cache/`에 종목별 pickle과 `meta.pkl`(종목 메타: 시장/상장폐지일/사유)로
저장된다(용량이 커서 git에는 올리지 않음). 이미 받은 종목은 건너뛰므로 중단 후 재실행해도 된다.

사용법: python scripts/build_pit_dataset.py [--limit N]   (N은 시험용으로 종목 수 제한)
"""
from __future__ import annotations

import io
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import _pyarrow_compat  # noqa: F401,E402

import FinanceDataReader as fdr  # noqa: E402
import pandas as pd  # noqa: E402
import requests  # noqa: E402

CACHE_DIR = ROOT / "data" / "pit_cache"
FETCH_START = "2014-03-01"  # 2016-01 백테스트 시작 + 13개월 모멘텀 워밍업을 충분히 덮는다
DELIST_MIN_DATE = pd.Timestamp("2015-06-01")  # 이보다 먼저 폐지된 종목은 백테스트 구간에 없다
DELIST_CACHE_API = "https://api.github.com/repos/FinanceData/fdr_krx_data_cache/contents/data/listing/delisting"
DELIST_RAW = "https://raw.githubusercontent.com/FinanceData/fdr_krx_data_cache/refs/heads/master/data/listing/delisting/{}"
MAX_WORKERS = 10
_PREFERRED = re.compile(r"\d?우[A-Z]?$")
_EXCLUDE_NAME = re.compile(r"스팩|SPAC|제\d+호")


def _is_common_stock_code(code: str) -> bool:
    """6자리 숫자이고 끝자리가 0인 보통주 코드만(우선주는 5/7/9/K 등으로 끝난다)."""
    return bool(re.fullmatch(r"\d{5}0", str(code)))


def load_listed() -> pd.DataFrame:
    listing = fdr.StockListing("KRX")
    listing = listing[listing["Market"].isin(("KOSPI", "KOSDAQ"))].copy()
    listing = listing[listing["Code"].map(_is_common_stock_code)]
    listing = listing[~listing["Name"].str.contains(_PREFERRED) & ~listing["Name"].str.contains(_EXCLUDE_NAME)]
    out = listing[["Code", "Name", "Market"]].copy()
    out["Shares"] = listing["Stocks"].astype(float)  # 현재 상장주식수(보정된 가격과 같은 기준)
    out["DelistingDate"] = pd.NaT
    out["Reason"] = None
    out["ToName"] = None
    return out


def load_delisted() -> pd.DataFrame:
    r = requests.get(DELIST_CACHE_API, timeout=30)
    r.raise_for_status()
    latest = sorted(x["name"] for x in r.json() if x["name"].endswith(".csv"))[-1]
    raw = requests.get(DELIST_RAW.format(latest), timeout=60)
    raw.raise_for_status()
    df = pd.read_csv(io.StringIO(raw.text), dtype={"Symbol": str})
    print(f"상장폐지 목록: {latest} 기준 {len(df)}건")
    df["DelistingDate"] = pd.to_datetime(df["DelistingDate"])
    df = df[(df["SecuGroup"] == "주권") & df["Market"].isin(("KOSPI", "KOSDAQ"))]
    df = df[df["Symbol"].map(_is_common_stock_code)]
    df = df[~df["Name"].str.contains(_PREFERRED) & ~df["Name"].str.contains(_EXCLUDE_NAME)]
    df = df[df["DelistingDate"] >= DELIST_MIN_DATE]
    df["Shares"] = df["ListingShares"].astype(float)  # 폐지 시점 상장주식수
    out = df.rename(columns={"Symbol": "Code"})[["Code", "Name", "Market", "Shares", "DelistingDate", "Reason", "ToName"]]
    return out.drop_duplicates("Code", keep="last")


def _fetch_one(code: str, end: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{code}.pkl"
    if path.exists():
        return None  # 이미 있음
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            df = fdr.DataReader(code, FETCH_START, end)
            df = df.dropna(subset=["Close"])
            df = df[df["Close"] > 0]
            with open(path, "wb") as f:
                pickle.dump(df, f)
            return df
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{type(last_err).__name__}: {last_err}")


def main() -> None:
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().date().isoformat()

    listed = load_listed()
    delisted = load_delisted()
    # 같은 코드가 두 목록에 있으면(재상장 등) 상장폐지 이력을 우선한다
    meta = pd.concat([delisted, listed[~listed["Code"].isin(delisted["Code"])]], ignore_index=True)
    print(f"현재 상장 보통주 {len(listed)}개 + 2015-06 이후 상장폐지 보통주 {len(delisted)}개 = {len(meta)}개")
    meta.to_pickle(CACHE_DIR / "meta.pkl")

    codes = meta["Code"].tolist()
    if limit:
        codes = codes[:limit]
    todo = [c for c in codes if not (CACHE_DIR / f"{c}.pkl").exists()]
    print(f"시세 조회 대상 {len(todo)}개 (이미 받은 {len(codes) - len(todo)}개는 건너뜀)\n")

    failed: list[tuple[str, str]] = []
    done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(_fetch_one, c, end): c for c in todo}
        for fut in as_completed(futs):
            code = futs[fut]
            try:
                fut.result()
            except Exception as e:  # noqa: BLE001
                failed.append((code, str(e)[:120]))
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(todo)} 완료 ({time.time() - t0:.0f}초, 실패 {len(failed)})", flush=True)

    print(f"\n완료: {len(todo) - len(failed)}개 저장, 실패 {len(failed)}개 ({time.time() - t0:.0f}초)")
    if failed:
        names = dict(zip(meta["Code"], meta["Name"]))
        for code, reason in failed[:40]:
            print(f"  ⚠️ {names.get(code, '?')}({code}): {reason}")
        with open(CACHE_DIR / "failed.pkl", "wb") as f:
            pickle.dump(failed, f)


if __name__ == "__main__":
    main()
