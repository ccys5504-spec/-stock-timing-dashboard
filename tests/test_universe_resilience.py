"""시가총액 상위 N개 선정(get_universe)이 KRX 전체 목록 조회 한 번에 너무 쉽게 무너지지 않는지 검증한다.

배경(2026-10-01): "오늘의 추천"을 100종목에서 300종목으로 넓히고 나서 Cloud에서 "데이터 조회 실패"로
전체 스캔이 한 번에 실패하는 걸 겪었다. 1차 원인은 get_universe()가 종목별이 아니라
fdr.StockListing("KRX") 단일 호출에 의존하는데, 그 호출이 캐싱도 재시도도 없이 바로 예외를 던지면
그대로 전체가 죽는 구조였다(재시도/캐시 공유를 추가해 1차 대응).

그런데 재시도를 추가한 뒤에도 Cloud 로그에서 실제 원인을 확인했다: fdr.StockListing("KRX")가
"오늘 날짜가 며칠이냐"만 물어보려고 호출하는 data.krx.co.kr의 작은 bld 엔드포인트가 Cloud에서
`ValueError: Failed to load data from http://data.krx.co.kr/...`로 매번(3번 재시도 전부) 실패했다
— 일시적 오류가 아니라 그 엔드포인트 자체가 막힌 것. 실제 종목 데이터는 원래도 GitHub 캐시
(FinanceData/fdr_krx_data_cache)에서 오므로, 그 "날짜 질문"을 data.krx.co.kr 대신 GitHub API로
해결하도록 바꿔서 이 엔드포인트를 아예 거치지 않게 했다(2차 대응, 이제 1순위 경로).
FDR의 기본 경로(fdr.StockListing)는 GitHub 캐시마저 안 될 때의 2순위 대비책으로 남겨뒀다.
"""
import pandas as pd
import pytest

import data.fetch as fetch


@pytest.fixture(autouse=True)
def _clear_listing_cache():
    fetch._full_krx_listing.cache_clear()
    yield
    fetch._full_krx_listing.cache_clear()


def _fake_listing() -> pd.DataFrame:
    return pd.DataFrame({
        "Code": ["000001", "000002", "000003"],
        "Name": ["가나다", "라마바우", "사아자"],
        "Market": ["KOSPI", "KOSPI", "KOSDAQ"],
        "Marcap": [300, 200, 100],
        "Dept": ["", "", ""],
    })


def _disable_github_cache(monkeypatch, error=RuntimeError("github 캐시 테스트에서 비활성화")):
    """1순위(GitHub 캐시) 경로를 끄고 2순위(fdr.StockListing) 경로만 테스트하기 위한 헬퍼."""
    def _raise():
        raise error
    monkeypatch.setattr(fetch, "_fetch_krx_listing_from_github_cache", _raise)


# ---- 1순위: GitHub 캐시 직접 조회 (data.krx.co.kr를 거치지 않음) -----------------------------

def test_github_cache_is_tried_first_and_fdr_is_not_called_when_it_succeeds(monkeypatch):
    fdr_calls = []
    monkeypatch.setattr(fetch.fdr, "StockListing", lambda market: fdr_calls.append(market) or _fake_listing())
    monkeypatch.setattr(fetch, "_fetch_krx_listing_from_github_cache", lambda: _fake_listing())
    out = fetch.get_universe(markets=("KOSPI", "KOSDAQ"), top_n=10)
    assert fdr_calls == []  # GitHub 캐시가 성공하면 data.krx.co.kr로 가는 FDR 경로는 아예 안 탄다
    assert "000002" not in out["Code"].tolist()  # 우선주(...우) 제외는 그대로 동작


def test_github_cache_reads_listing_then_latest_csv(monkeypatch):
    requested = []

    class FakeResponse:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self._payload
        @property
        def text(self):
            return self._payload

    def fake_get(url, timeout=None):
        requested.append(url)
        if url == fetch._KRX_LISTING_CACHE_API:
            return FakeResponse([{"name": "2026-09-30.csv"}, {"name": "2026-10-01.csv"}, {"name": "README.md"}])
        assert url == fetch._KRX_LISTING_CACHE_RAW.format("2026-10-01.csv")  # 날짜순으로 가장 최신 파일을 고른다
        return FakeResponse(_fake_listing().to_csv())

    monkeypatch.setattr(fetch.requests, "get", fake_get)
    out = fetch._fetch_krx_listing_from_github_cache()
    assert len(requested) == 2
    assert set(out["Code"]) == {"000001", "000002", "000003"}


# ---- 2순위: GitHub 캐시가 실패하면 fdr.StockListing(+재시도)로 넘어간다 ------------------------

def test_get_universe_reuses_the_cached_listing_instead_of_fetching_again(monkeypatch):
    _disable_github_cache(monkeypatch)
    calls = []

    def fake_stock_listing(market):
        calls.append(market)
        return _fake_listing()

    monkeypatch.setattr(fetch.fdr, "StockListing", fake_stock_listing)
    fetch.resolve_stock_name("000001")  # 먼저 이 경로로 한 번 캐시를 채운다
    out = fetch.get_universe(markets=("KOSPI", "KOSDAQ"), top_n=10)
    assert calls == ["KRX"]  # get_universe가 따로 또 부르지 않고 캐시를 재사용했다
    assert "000002" not in out["Code"].tolist()  # 우선주(...우) 제외는 그대로 동작


def test_listing_fetch_retries_transient_failures_then_succeeds(monkeypatch):
    _disable_github_cache(monkeypatch)
    calls = {"n": 0}
    sleeps = []

    def flaky(market):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("일시적 네트워크 오류")
        return _fake_listing()

    monkeypatch.setattr(fetch.fdr, "StockListing", flaky)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: sleeps.append(s))
    out = fetch.get_universe(markets=("KOSPI", "KOSDAQ"), top_n=10)
    assert calls["n"] == 3  # 2번 실패 후 3번째에 성공
    assert len(sleeps) == 2  # 실패할 때마다 짧게 쉬었다 재시도
    assert len(out) >= 1


def test_listing_fetch_raises_original_error_after_exhausting_retries(monkeypatch):
    _disable_github_cache(monkeypatch)
    calls = {"n": 0}

    def always_fails(market):
        calls["n"] += 1
        raise ConnectionError("KRX 서버 응답 없음")

    monkeypatch.setattr(fetch.fdr, "StockListing", always_fails)
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    with pytest.raises(ConnectionError, match="KRX 서버 응답 없음"):
        fetch.get_universe(markets=("KOSPI", "KOSDAQ"), top_n=10)
    assert calls["n"] == 3  # 재시도 3번 다 쓰고 나서야 포기한다

    # lru_cache는 예외를 기억하지 않으므로, 다음 호출에서 다시 처음부터 시도한다
    # (resolve_stock_name은 실패를 삼키고 None을 돌려주는 쪽이라 그 경로로 "다시 시도했는지"를 확인한다)
    assert fetch.resolve_stock_name("000001") is None
    assert calls["n"] == 6  # 두 번째 요청도 다시 3번을 전부 시도했다(이전 실패가 캐싱되지 않음)
