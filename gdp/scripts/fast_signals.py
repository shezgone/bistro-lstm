"""E1-lite 빠른신호: 일별 금융(KOSPI·원달러, yfinance — 가격은 무개정이라 vintage-safe)
+ 원시 빈티지의 ESI 원지수. 월별 시계열로 집계해 TSFM 공변량/컨포멀 조건 변수로 사용.

주의: NSI(한은 뉴스심리지수)·관세청 10일 수출은 API 키 필요 → 차기(액션 아이템).
"""
import os, glob
import numpy as np, pandas as pd

_CACHE = "data/fast_signals_daily.parquet"
_raw_cache = {}

def daily_market(refresh=False):
    """일별 KOSPI 종가·원달러 환율 (2000~). 캐시 사용."""
    if os.path.exists(_CACHE) and not refresh:
        return pd.read_parquet(_CACHE)
    import yfinance as yf
    px = yf.download(["^KS11", "KRW=X"], start="2000-01-01", progress=False)["Close"]
    px.columns = ["krw", "kospi"] if list(px.columns) == ["KRW=X", "^KS11"] else ["kospi", "krw"]
    px = px.rename(columns={"^KS11": "kospi", "KRW=X": "krw"})
    px.to_parquet(_CACHE)
    return px

def load_raw_vintage(vintage):
    files = sorted(glob.glob("data/vintages/*.xlsx"))
    dates = [os.path.basename(f)[:-5] for f in files]
    cands = [d for d in dates if d <= vintage]
    if not cands: return None
    f = cands[-1]
    if f not in _raw_cache:
        x = pd.read_excel(f"data/vintages/{f}.xlsx")
        if "date" in x.columns: x = x.rename(columns={"date": "Date"})
        x["Date"] = pd.to_datetime(x["Date"])
        _raw_cache[f] = x.set_index("Date")
    return _raw_cache[f]

def monthly_covariates(month_index, vintage):
    """month_index(월말 DatetimeIndex)에 정렬된 빠른신호 4종.
    일별 데이터는 vintage 날짜까지만 사용(현재 미완성 월은 부분 수익률)."""
    v = pd.Timestamp(vintage)
    px = daily_market()
    px = px[px.index <= v]
    out = pd.DataFrame(index=month_index, dtype=np.float32)
    me = px.resample("ME").last().reindex(month_index)
    for c in ["kospi", "krw"]:
        ret = me[c].pct_change().astype(np.float32)
        out[f"{c}_mret"] = ret.fillna(0.0).clip(-0.3, 0.3)
    raw = load_raw_vintage(vintage)
    if raw is not None and "new_esi" in raw.columns:
        esi = raw["new_esi"].reindex(month_index).astype(np.float32)
        out["esi_raw"] = esi.ffill().fillna(100.0)
        out["esi_mom"] = (out["esi_raw"] - out["esi_raw"].rolling(4).mean()).fillna(0.0)
    else:
        out["esi_raw"] = 100.0; out["esi_mom"] = 0.0
    return out
