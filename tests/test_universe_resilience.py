"""시가총액 상위 N개 선정(get_universe)이 KRX 전체 목록 조회 한 번에 너무 쉽게 무너지지 않는지 검증한다.

배경(2026-10-01): "오늘의 추천"을 100종목에서 300종목으로 넓히고 나서 Cloud에서 "데이터 조회 실패"로
전체 스캔이 한 번에 실패하는 걸 겪었다. 원인은 get_universe()가 종목별이 아니라 fdr.StockListing("KRX")
단일 호출에 의존하는데, 그 호출이 캐싱도 재시도도 없이 바로 예외를 던지면 그대로 전체가 죽는 구조였다.
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


def test_get_universe_reuses_the_cached_listing_instead_of_fetching_again(monkeypatch):
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
