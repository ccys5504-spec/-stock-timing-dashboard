"""보유종목/설정 저장소 — 테스트가 실제 Gist나 실제 data/*.json을 건드리지 않도록
경로와 캐시를 임시 폴더로 돌려놓고 검증한다."""
import json

import pytest

import data.fetch as fetch


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "_WATCHLIST_PATH", tmp_path / "watchlist.json")
    monkeypatch.setattr(fetch, "_HOLDINGS_PATH", tmp_path / "holdings.json")
    monkeypatch.setattr(fetch, "_SETTINGS_PATH", tmp_path / "settings.json")
    for cache in ("_watchlist_cache", "_holdings_cache", "_settings_cache"):
        monkeypatch.setattr(fetch, cache, None)
    # 어떤 경우에도 네트워크(Gist)로 나가지 못하게 막는다
    monkeypatch.setattr(fetch, "_gist_config", lambda: None)
    monkeypatch.setattr(fetch, "_gist_read_file", lambda filename: None)
    monkeypatch.setattr(fetch, "_gist_write_file", lambda filename, content: False)
    return tmp_path


def test_tests_never_see_real_secrets():
    assert not fetch._SECRETS_PATH.exists()
    assert fetch._gist_config() is None


def test_holdings_roundtrip_via_local_file(isolated_store):
    assert fetch.load_holdings() == []
    holdings = [{"종목코드": "005930", "수량": 10, "매입단가": 70000}]
    fetch.save_holdings(holdings)
    assert json.loads((isolated_store / "holdings.json").read_text(encoding="utf-8")) == holdings
    # 캐시를 비우고 파일에서 다시 읽어도 같아야 한다
    fetch._holdings_cache = None
    assert fetch.load_holdings() == holdings


def test_settings_fill_missing_keys_with_defaults(isolated_store):
    (isolated_store / "settings.json").write_text(json.dumps({"threshold": 2}), encoding="utf-8")
    s = fetch.load_settings()
    assert s["threshold"] == 2
    assert s["volume_multiplier"] == fetch._DEFAULT_SETTINGS["volume_multiplier"]


def test_settings_save_then_load_returns_saved_values(isolated_store):
    fetch.save_settings({**fetch._DEFAULT_SETTINGS, "threshold": 2, "use_stop_loss": True})
    fetch._settings_cache = None
    s = fetch.load_settings()
    assert s["threshold"] == 2 and s["use_stop_loss"] is True


def test_watchlist_defaults_created_when_missing(isolated_store):
    wl = fetch.load_watchlist()
    assert wl == fetch._DEFAULT_WATCHLIST
    assert (isolated_store / "watchlist.json").exists()


def test_gist_used_when_configured(isolated_store, monkeypatch):
    written = {}
    monkeypatch.setattr(fetch, "_gist_read_file", lambda filename: json.dumps([{"종목코드": "000660"}]))
    monkeypatch.setattr(fetch, "_gist_write_file", lambda filename, content: written.update({filename: content}) or True)
    assert fetch.load_holdings() == [{"종목코드": "000660"}]
    fetch.save_holdings([{"종목코드": "035420"}])
    assert "holdings.json" in written
    assert not (isolated_store / "holdings.json").exists()  # Gist 저장 성공 시 로컬 파일은 쓰지 않는다


def test_atomic_write_leaves_no_temp_files(tmp_path):
    target = tmp_path / "x.json"
    fetch._atomic_write_json(target, {"a": 1})
    fetch._atomic_write_json(target, {"a": 2})
    assert json.loads(target.read_text(encoding="utf-8")) == {"a": 2}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
