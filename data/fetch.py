"""국내 주식 일봉 시세 수집 모듈."""
from __future__ import annotations

import datetime as dt
import functools
import json
import os
import tempfile
import time
import tomllib
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import requests

_WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"
_HOLDINGS_PATH = Path(__file__).parent / "holdings.json"

# ---- 관심종목/보유종목 저장 위치 ----
# 2026-09-16: Streamlit Cloud는 코드가 재배포될 때마다(새 push든 수동
# Reboot든) 컨테이너를 완전히 새로 만든다. data/*.json은 개인정보라 git에
# 올리지 않는데(.gitignore), 바로 그 이유 때문에 재배포마다 Cloud에서
# 직접 입력한 보유종목/관심종목이 통째로 사라지는 사고가 있었다.
#
# 그래서 GitHub Gist(비공개)에 저장해서 재배포와 무관하게 남도록 바꿨다.
# `.streamlit/secrets.toml`에 GITHUB_TOKEN/GIST_ID가 설정돼 있으면 그
# Gist를 읽고 쓰며, 없으면(로컬 개발 등) 예전처럼 로컬 JSON 파일로
# 동작한다 — 둘 다 안 되면 앱이 아예 안 켜지는 일은 없도록 항상 로컬
# 파일을 최종 대비책으로 둔다.
_SECRETS_PATH = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
_GIST_API_TIMEOUT = 10
_GIST_CACHE_TTL = 10  # 초 — 매 rerun마다 Gist API를 부르지 않도록 짧게 캐싱


@functools.lru_cache(maxsize=1)
def _load_secrets() -> dict:
    if not _SECRETS_PATH.exists():
        return {}
    try:
        with open(_SECRETS_PATH, "rb") as f:
            return tomllib.load(f)
    except Exception:  # noqa: BLE001
        return {}


def _gist_config() -> tuple[str, str] | None:
    secrets = _load_secrets()
    token = secrets.get("GITHUB_TOKEN")
    gist_id = secrets.get("GIST_ID")
    if token and gist_id:
        return token, gist_id
    return None


def _gist_headers(token: str) -> dict:
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}


def _gist_read_file(filename: str) -> str | None:
    """Gist가 설정돼 있으면 그 안의 filename 내용을 문자열로 반환한다.
    설정이 없거나 그 파일이 아직 없거나 API 호출이 실패하면 None —
    호출부에서 로컬 파일로 대체(fallback)한다."""
    config = _gist_config()
    if config is None:
        return None
    token, gist_id = config
    try:
        resp = requests.get(
            f"https://api.github.com/gists/{gist_id}",
            headers=_gist_headers(token), timeout=_GIST_API_TIMEOUT,
        )
        resp.raise_for_status()
        files = resp.json().get("files", {})
    except Exception:  # noqa: BLE001
        return None
    file_info = files.get(filename)
    return file_info["content"] if file_info else None


def _gist_write_file(filename: str, content: str) -> bool:
    """Gist가 설정돼 있으면 그 안의 filename을 갱신한다. 성공하면 True."""
    config = _gist_config()
    if config is None:
        return False
    token, gist_id = config
    try:
        resp = requests.patch(
            f"https://api.github.com/gists/{gist_id}",
            headers=_gist_headers(token),
            json={"files": {filename: {"content": content}}},
            timeout=_GIST_API_TIMEOUT,
        )
        resp.raise_for_status()
        return True
    except Exception:  # noqa: BLE001
        return False


def _atomic_write_json(path: Path, data) -> None:
    """파일이 깨지지 않도록 안전하게 JSON을 저장한다.

    이전에는 open(path, "w")로 바로 덮어썼는데, 그 방식은 (1) 저장 도중
    프로그램이 죽거나 (2) 브라우저 탭 두 개에서 거의 동시에 저장 버튼을
    누르는 경우, 파일이 절반만 써진 채로 남거나 서로 다른 내용이 뒤섞일 수
    있다. 대신 같은 폴더에 임시 파일로 전체 내용을 먼저 다 쓰고, 그 다음
    os.replace()로 한 번에 교체한다 — os.replace는 운영체제 수준에서
    "원자적"이라서, 중간에 어떤 일이 생겨도 원본 파일은 '이전 내용 그대로'
    이거나 '새 내용으로 완전히 바뀐 상태' 둘 중 하나만 존재하고 깨진 상태로
    남지 않는다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

_DEFAULT_WATCHLIST: dict[str, str] = {
    "SK하이닉스": "000660",
    "두산에너빌리티": "034020",
    "주성엔지니어링": "036930",
}

_watchlist_cache: dict[str, str] | None = None
_watchlist_cache_at: float = 0.0
_holdings_cache: list[dict] | None = None
_holdings_cache_at: float = 0.0


def load_watchlist() -> dict[str, str]:
    """관심 종목(종목명 -> 종목코드)을 읽어온다.

    Gist가 설정돼 있으면 Gist에서, 아니면 data/watchlist.json에서 읽는다
    (둘 다 없으면 기본값으로 새로 만든다). 매 Streamlit rerun마다 다시
    호출되므로, Gist 사용 시 짧게(10초) 캐싱해서 API를 과도하게 부르지
    않는다 — save_watchlist() 직후에는 캐시를 바로 최신값으로 갱신해서
    방금 저장한 내용이 곧바로 반영되게 한다.
    """
    global _watchlist_cache, _watchlist_cache_at
    now = time.monotonic()
    if _watchlist_cache is not None and now - _watchlist_cache_at < _GIST_CACHE_TTL:
        return dict(_watchlist_cache)

    content = _gist_read_file("watchlist.json")
    if content is not None:
        result = json.loads(content)
    elif _WATCHLIST_PATH.exists():
        with open(_WATCHLIST_PATH, encoding="utf-8") as f:
            result = json.load(f)
    else:
        save_watchlist(_DEFAULT_WATCHLIST)
        return dict(_DEFAULT_WATCHLIST)

    _watchlist_cache, _watchlist_cache_at = result, now
    return dict(result)


def save_watchlist(watchlist: dict[str, str]) -> None:
    """관심 종목을 저장한다 (Gist가 설정돼 있으면 Gist에, 아니면 로컬 파일에)."""
    global _watchlist_cache, _watchlist_cache_at
    if not _gist_write_file("watchlist.json", json.dumps(watchlist, ensure_ascii=False, indent=2)):
        _atomic_write_json(_WATCHLIST_PATH, watchlist)
    _watchlist_cache, _watchlist_cache_at = dict(watchlist), time.monotonic()


# 관심 종목 (종목명 -> 종목코드). 앱에서 교체하면 저장되고, 다음 실행부터
# 반영된다. 이번 세션 안에서 즉시 반영하려면 load_watchlist()를 다시
# 호출해야 한다(app.py에서 그렇게 처리함).
WATCHLIST: dict[str, str] = load_watchlist()


def load_holdings() -> list[dict]:
    """내 보유종목 목록을 읽어온다.

    각 항목: {"종목코드": str, "수량": float, "매입단가": float}
    Gist가 설정돼 있으면 Gist에서, 아니면 data/holdings.json에서 읽는다.
    둘 다 없으면 빈 목록(기본으로 보유종목을 가정하지 않음).
    """
    global _holdings_cache, _holdings_cache_at
    now = time.monotonic()
    if _holdings_cache is not None and now - _holdings_cache_at < _GIST_CACHE_TTL:
        return list(_holdings_cache)

    content = _gist_read_file("holdings.json")
    if content is not None:
        result = json.loads(content)
    elif _HOLDINGS_PATH.exists():
        with open(_HOLDINGS_PATH, encoding="utf-8") as f:
            result = json.load(f)
    else:
        result = []

    _holdings_cache, _holdings_cache_at = result, now
    return list(result)


def save_holdings(holdings: list[dict]) -> None:
    """내 보유종목 목록을 저장한다 (Gist가 설정돼 있으면 Gist에, 아니면 로컬 파일에)."""
    global _holdings_cache, _holdings_cache_at
    if not _gist_write_file("holdings.json", json.dumps(holdings, ensure_ascii=False, indent=2)):
        _atomic_write_json(_HOLDINGS_PATH, holdings)
    _holdings_cache, _holdings_cache_at = list(holdings), time.monotonic()


@functools.lru_cache(maxsize=1)
def _full_krx_listing() -> pd.DataFrame:
    """전체 KRX 종목 목록(코드 -> 종목명 조회용)을 한 번만 받아와 캐싱한다."""
    return fdr.StockListing("KRX")


def resolve_stock_name(code: str) -> str | None:
    """종목코드로 종목명을 찾는다. 못 찾거나 조회에 실패하면 None."""
    try:
        listing = _full_krx_listing()
        match = listing[listing["Code"] == code]
    except Exception:  # noqa: BLE001
        return None
    if match.empty:
        return None
    return str(match.iloc[0]["Name"])


def fetch_ohlcv(code: str, years: int = 3) -> pd.DataFrame:
    """종목코드로 최근 N년치 일봉 OHLCV 데이터를 가져온다.

    Returns columns: Open, High, Low, Close, Volume (index: 날짜)

    2026-09-15 수정: 예전에는 여기서 300일치 워밍업을 직접 더한 start를 만든
    뒤 fetch_ohlcv_range()에 넘겼는데, 그 함수가 내부적으로 또 warmup_days
    (기본 300일)를 더하고 있어서 실제로는 워밍업이 600일(약 300+300)로
    이중으로 붙고 있었다. 결과가 틀리지는 않지만(워밍업은 많을수록 안전한
    방향) 필요 이상으로 더 오래된 데이터까지 매번 불필요하게 내려받아서
    조회가 느려지고, 코드만 봐서는 "왜 300일이 아니라 600일이지?"가 헷갈렸다.
    이제 여기서는 순수하게 "표시하고 싶은 기간"의 시작일만 계산하고, 워밍업은
    fetch_ohlcv_range()에게 한 번만 맡긴다.
    """
    end = dt.date.today()
    start = end - dt.timedelta(days=int(years * 365.25))
    return fetch_ohlcv_range(code, start.isoformat(), end.isoformat())


def get_universe(
    markets: tuple[str, ...] = ("KOSPI", "KOSDAQ"),
    top_n: int = 50,
) -> pd.DataFrame:
    """시가총액 상위 N개 종목 목록을 가져온다 (우선주·관리종목·스팩 등은 제외).

    Returns columns: Code, Name, Market, Marcap
    """
    import re

    listing = fdr.StockListing("KRX")
    listing = listing[listing["Market"].isin(markets)].copy()

    # 우선주 제외 (종목명이 '...우', '...우B', '...2우B' 등으로 끝남)
    is_preferred = listing["Name"].str.contains(r"\d?우[A-Z]?$", regex=True)
    listing = listing[~is_preferred]

    # 관리종목/투자주의환기종목/스팩 등 제외
    exclude_keywords = "관리종목|투자주의|SPAC|스팩"
    if "Dept" in listing.columns:
        listing = listing[~listing["Dept"].astype(str).str.contains(exclude_keywords, regex=True, na=False)]

    listing = listing.sort_values("Marcap", ascending=False).head(top_n)
    return listing[["Code", "Name", "Market", "Marcap"]].reset_index(drop=True)


def fetch_ohlcv_range(code: str, start: str, end: str, warmup_days: int = 300) -> pd.DataFrame:
    """종목코드로 [start, end] 구간의 일봉 OHLCV 데이터를 가져온다.

    이평선(MA60), 상승장/하락장 자동판별(200일 이평선) 등 지표 계산에 필요한
    워밍업 구간(기본 300일)만큼 start 이전 데이터도 함께 가져온다 — 신호/백테스트
    코드에서 실제 표시 구간만 다시 잘라서 쓰면 된다.

    start/end: "YYYY-MM-DD" 형식 문자열
    """
    start_date = dt.date.fromisoformat(start) - dt.timedelta(days=warmup_days)
    df = fdr.DataReader(code, start_date.isoformat(), end)
    if df.empty:
        raise ValueError(f"'{code}' 종목의 데이터를 가져오지 못했습니다.")
    df = df.dropna(subset=["Close"]).copy()
    return df
