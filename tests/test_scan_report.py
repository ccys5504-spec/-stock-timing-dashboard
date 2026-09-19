"""스캐너가 실패 종목을 조용히 버리지 않고 attrs로 보고하는지 검증."""
import pandas as pd

import screener.scan as scan
from tests.helpers import make_ohlcv


def _candidates(codes):
    return pd.DataFrame({
        "Code": codes, "Name": [f"종목{c}" for c in codes],
        "Market": ["KOSPI"] * len(codes), "Marcap": [1e12] * len(codes),
    })


def test_failed_and_short_stocks_are_reported(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=300)

    def fake_fetch(code, years=1):
        if code == "BAD":
            raise ConnectionError("timeout")
        if code == "NEW":
            return make_ohlcv(dates[-20:], seed=3)  # 상장 직후: 지표를 못 만든다
        return make_ohlcv(dates, seed=1)

    monkeypatch.setattr(scan, "fetch_ohlcv", fake_fetch)
    result = scan.scan_signals(_candidates(["OK1", "OK2", "BAD", "NEW"]), max_workers=2)

    assert result.attrs["requested"] == 4
    failed = {code: reason for code, _name, reason in result.attrs["failed"]}
    assert set(failed) == {"BAD", "NEW"}
    assert "ConnectionError" in failed["BAD"]
    assert len(result) == 2
    assert set(result["종목코드"]) == {"OK1", "OK2"}


def test_all_failed_still_returns_report(monkeypatch):
    def boom(code, years=1):
        raise RuntimeError("down")

    monkeypatch.setattr(scan, "fetch_ohlcv", boom)
    result = scan.scan_signals(_candidates(["A", "B"]), max_workers=2)
    assert result.empty
    assert result.attrs["requested"] == 2
    assert len(result.attrs["failed"]) == 2
