"""매매 기록(일지)으로 보유종목·실현손익·일일 수익률을 계산한다 — 화면(views/journal.py)이 사용한다.

이 앱은 증권사와 연동되어 있지 않아서, 체결 내역은 사용자가 직접 입력한 기록만 안다.
기록 한 건: {"날짜": "YYYY-MM-DD", "종목코드": "005930", "구분": "매수"|"매도",
             "수량": 10, "단가": 70000, "수수료세금": 0}   (수수료세금은 원 단위 합계, 생략 가능)

계산 방식(모두 기록과 종가만 쓴다):
  - 평균단가법: 매수하면 평균단가가 가중평균으로 바뀌고, 매도하면 평균단가는 그대로 두고 수량만 줄인다.
  - 실현손익 = (매도가 - 평균단가) x 수량 - 매도 수수료세금(매수 수수료는 평균단가에 포함).
  - 일일 수익률(자금 유출입을 반영한 근사, 수정 디에츠 방식):
      일손익 = 오늘 종가 평가액 + 오늘 매도대금 - 오늘 매수금액 - 어제 종가 평가액
      일수익률 = 일손익 / (어제 종가 평가액 + 오늘 매수금액)
    오늘 산 종목은 매수가 대비 종가로 그날 손익이 잡힌다. 누적 수익률은 일수익률을 곱해서 잇는다
    (시간가중 수익률: 언제 얼마를 넣었는지와 무관하게 "운용 성과"를 본다).
  - 보유 종목이 하나도 없는 날은 수익률 계산에서 제외(0으로 이어 붙임).
"""
from __future__ import annotations

import pandas as pd

BUY, SELL = "매수", "매도"


class JournalError(ValueError):
    """기록이 앞뒤가 안 맞을 때(보유보다 많이 매도 등)."""


def normalize_trades(trades: list[dict]) -> pd.DataFrame:
    """기록 목록을 날짜순 표로 정리하고 형식을 검증한다. 같은 날 여러 건은 입력 순서를 유지한다."""
    rows = []
    for i, t in enumerate(trades):
        try:
            date = pd.Timestamp(t["날짜"]).normalize()
            code = str(t["종목코드"]).strip().zfill(6)
            side = str(t["구분"]).strip()
            qty = float(t["수량"])
            price = float(t["단가"])
            fee = float(t.get("수수료세금") or 0)
        except (KeyError, TypeError, ValueError) as e:
            raise JournalError(f"{i + 1}번째 기록을 읽을 수 없습니다: {e}") from e
        if side not in (BUY, SELL):
            raise JournalError(f"{i + 1}번째 기록의 구분이 '매수'/'매도'가 아닙니다: {side}")
        if qty <= 0 or price <= 0 or fee < 0:
            raise JournalError(f"{i + 1}번째 기록의 수량·단가는 0보다 커야 하고 수수료는 0 이상이어야 합니다.")
        rows.append({"날짜": date, "종목코드": code, "구분": side, "수량": qty, "단가": price,
                     "수수료세금": fee, "_순번": i})
    df = pd.DataFrame(rows, columns=["날짜", "종목코드", "구분", "수량", "단가", "수수료세금", "_순번"])
    return df.sort_values(["날짜", "_순번"], kind="stable").reset_index(drop=True)


def apply_trade_to_holdings(holdings: list[dict], trade: dict) -> list[dict]:
    """보유종목 목록(종목코드/수량/매입단가)에 기록 한 건을 반영한 새 목록을 돌려준다(평균단가법)."""
    t = normalize_trades([trade]).iloc[0]
    out = [dict(h) for h in holdings]
    idx = next((i for i, h in enumerate(out) if str(h.get("종목코드", "")).strip().zfill(6) == t["종목코드"]), None)
    if t["구분"] == BUY:
        cost = t["수량"] * t["단가"] + t["수수료세금"]
        if idx is None:
            out.append({"종목코드": t["종목코드"], "수량": t["수량"], "매입단가": round(cost / t["수량"], 2)})
        else:
            q0, p0 = float(out[idx].get("수량") or 0), float(out[idx].get("매입단가") or 0)
            q1 = q0 + t["수량"]
            out[idx]["수량"] = q1
            out[idx]["매입단가"] = round((q0 * p0 + cost) / q1, 2)
        return out
    if idx is None:
        raise JournalError(f"{t['종목코드']}은(는) 보유종목에 없어서 매도를 반영할 수 없습니다.")
    q0 = float(out[idx].get("수량") or 0)
    if t["수량"] > q0 + 1e-9:
        raise JournalError(f"{t['종목코드']} 보유 수량({q0:g}주)보다 많이({t['수량']:g}주) 매도할 수 없습니다.")
    remaining = q0 - t["수량"]
    if remaining <= 1e-9:
        del out[idx]
    else:
        out[idx]["수량"] = remaining
    return out


def realized_pnl(trades: list[dict]) -> pd.DataFrame:
    """매도 한 건마다의 실현손익(평균단가법). 열: 날짜, 종목코드, 수량, 매도가, 평균단가, 실현손익(원), 수익률."""
    df = normalize_trades(trades)
    pos: dict[str, tuple[float, float]] = {}  # 코드 -> (수량, 평균단가)
    rows = []
    for t in df.itertuples():
        qty0, avg0 = pos.get(t.종목코드, (0.0, 0.0))
        if t.구분 == BUY:
            cost = t.수량 * t.단가 + t.수수료세금
            qty1 = qty0 + t.수량
            pos[t.종목코드] = (qty1, (qty0 * avg0 + cost) / qty1)
            continue
        if t.수량 > qty0 + 1e-9:
            raise JournalError(f"{t.날짜.date()} {t.종목코드}: 보유 수량({qty0:g}주)보다 많이 매도했습니다.")
        pnl = (t.단가 - avg0) * t.수량 - t.수수료세금
        rows.append({
            "날짜": t.날짜, "종목코드": t.종목코드, "수량": t.수량, "매도가": t.단가, "평균단가": avg0,
            "실현손익(원)": pnl, "수익률": pnl / (avg0 * t.수량) if avg0 > 0 else float("nan"),
        })
        pos[t.종목코드] = (qty0 - t.수량, avg0)
    return pd.DataFrame(rows, columns=["날짜", "종목코드", "수량", "매도가", "평균단가", "실현손익(원)", "수익률"])


def daily_performance(trades: list[dict], closes: dict[str, pd.Series]) -> pd.DataFrame:
    """일별 평가액·손익·수익률. closes는 {종목코드: 종가 시리즈(날짜 인덱스)}.

    열: 평가금액, 매수금액, 매도대금, 일손익(원), 일수익률, 누적수익률. 인덱스는 첫 거래일부터 마지막 종가일까지의
    종가 날짜. 시세가 없는 종목/날짜는 직전 종가로 평가한다.
    """
    df = normalize_trades(trades)
    if df.empty:
        return pd.DataFrame(columns=["평가금액", "매수금액", "매도대금", "일손익(원)", "일수익률", "누적수익률"])
    missing = sorted(set(df["종목코드"]) - set(closes))
    if missing:
        raise JournalError("시세를 받지 못한 종목이 있어 수익률을 계산할 수 없습니다: " + ", ".join(missing))

    start = df["날짜"].min()
    calendar = sorted({d for s in closes.values() for d in s.index if d >= start})
    if not calendar:
        return pd.DataFrame(columns=["평가금액", "매수금액", "매도대금", "일손익(원)", "일수익률", "누적수익률"])
    # 첫 거래일이 휴장일이면 그날을 달력에 넣어 거래 흐름이 빠지지 않게 한다
    calendar = sorted(set(calendar) | set(df["날짜"]))
    close_df = pd.DataFrame({c: s for c, s in closes.items() if c in set(df["종목코드"])}).reindex(calendar).ffill()

    qty: dict[str, float] = {}
    by_day = {d: g for d, g in df.groupby("날짜")}
    prev_value = 0.0
    rows = []
    cum = 1.0
    for date in calendar:
        buys = sells = 0.0
        for t in by_day.get(date, pd.DataFrame()).itertuples():
            if t.구분 == BUY:
                qty[t.종목코드] = qty.get(t.종목코드, 0.0) + t.수량
                buys += t.수량 * t.단가 + t.수수료세금
            else:
                held = qty.get(t.종목코드, 0.0)
                if t.수량 > held + 1e-9:
                    raise JournalError(f"{date.date()} {t.종목코드}: 보유 수량({held:g}주)보다 많이 매도했습니다.")
                qty[t.종목코드] = held - t.수량
                sells += t.수량 * t.단가 - t.수수료세금
        value = 0.0
        for code, q in qty.items():
            if q > 1e-9:
                px = close_df.at[date, code]
                if pd.isna(px):
                    raise JournalError(f"{date.date()} {code}의 종가를 찾을 수 없습니다.")
                value += q * float(px)
        pnl = value + sells - buys - prev_value
        base = prev_value + buys
        ret = pnl / base if base > 0 else 0.0
        cum *= 1 + ret
        rows.append({"날짜": date, "평가금액": value, "매수금액": buys, "매도대금": sells,
                     "일손익(원)": pnl, "일수익률": ret, "누적수익률": cum - 1})
        prev_value = value
    return pd.DataFrame(rows).set_index("날짜")


def summarize(trades: list[dict], perf: pd.DataFrame) -> dict:
    """화면 상단 요약: 실현손익 합, 평가손익(=총손익-실현), 총손익, 누적(시간가중) 수익률."""
    realized = realized_pnl(trades)["실현손익(원)"].sum() if trades else 0.0
    total = float(perf["일손익(원)"].sum()) if not perf.empty else 0.0
    return {
        "realized": float(realized), "total_pnl": total, "unrealized": total - float(realized),
        "cum_return": float(perf["누적수익률"].iloc[-1]) if not perf.empty else 0.0,
    }


def journal_qty(trades: list[dict], code: str) -> float:
    """매매 기록상 이 종목의 현재 보유 수량(기록이 없으면 0)."""
    code = str(code).strip().zfill(6)
    qty = 0.0
    for t in trades:
        if str(t.get("종목코드", "")).strip().zfill(6) == code:
            qty += float(t.get("수량") or 0) * (1 if t.get("구분") == BUY else -1)
    return qty


def opening_buy_for_sell(trades: list[dict], holdings: list[dict], code: str, qty: float, on_date: str) -> dict | None:
    """기록에 매수 이력이 없는 종목을 매도할 때, 모자란 수량만큼을 보유종목의 매입단가로 '기초 매수'로 남긴다.

    이미 보유 중이던 종목(이 메뉴를 쓰기 전에 산 것)을 처음 매도해도 실현손익·수익률이 계산되게 하려는 용도다.
    필요 없으면 None. 주의: 보유종목의 매입단가는 이후 추가 매수분까지 섞인 평균이라 근사치다.
    """
    code = str(code).strip().zfill(6)
    shortfall = qty - journal_qty(trades, code)
    if shortfall <= 1e-9:
        return None
    h = next((x for x in holdings if str(x.get("종목코드", "")).strip().zfill(6) == code), None)
    cost = float(h.get("매입단가") or 0) if h else 0.0
    if cost <= 0:
        return None
    return {"날짜": on_date, "종목코드": code, "구분": BUY, "수량": shortfall, "단가": cost, "수수료세금": 0.0}
