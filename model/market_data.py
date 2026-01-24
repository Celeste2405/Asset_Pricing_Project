from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List
import datetime as dt

import numpy as np
import pandas as pd


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class OptionQuote:
    ticker: str
    expiry: str
    option_type: OptionType
    strike: float
    last_price: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    mid: Optional[float]
    volume: Optional[float]
    open_interest: Optional[float]
    implied_vol: Optional[float]


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        val = float(x)
        if np.isnan(val):
            return None
        return val
    except Exception:
        return None


def mid_price(bid: Optional[float], ask: Optional[float], last: Optional[float]) -> Optional[float]:
    """
    Règle simple :
    - si bid/ask valides et ask>=bid>0 → mid = (bid+ask)/2
    - sinon fallback last si >0
    - sinon None
    """
    bid = _safe_float(bid)
    ask = _safe_float(ask)
    last = _safe_float(last)

    if bid is not None and ask is not None and bid > 0 and ask > 0 and ask >= bid:
        return 0.5 * (bid + ask)
    if last is not None and last > 0:
        return last
    return None


def year_fraction(expiry: str, asof: Optional[dt.date] = None) -> float:
    """
    expiry: 'YYYY-MM-DD'
    Convention simple ACT/365.
    """
    if asof is None:
        asof = dt.date.today()
    y, m, d = map(int, expiry.split("-"))
    exp = dt.date(y, m, d)
    days = (exp - asof).days
    return max(days / 365.0, 0.0)


def list_expiries(ticker: str) -> List[str]:
    """
    Retourne la liste des maturités disponibles sur Yahoo.
    """
    import yfinance as yf  # import local pour éviter de casser tout le projet si non installé

    tk = yf.Ticker(ticker)
    exps = getattr(tk, "options", None)
    return list(exps) if exps is not None else []


def get_option_chain(
    ticker: str,
    expiry: str,
    option_type: Optional[OptionType] = None,
    min_volume: int = 0,
    min_open_interest: int = 0,
    require_bid_ask: bool = False,
) -> pd.DataFrame:
    """
    Récupère la chaîne d'options via yfinance et renvoie un DataFrame standardisé :
        columns: ['ticker','expiry','type','K','last','bid','ask','mid','volume','openInterest','impliedVolatility','T']
    Filtrage simple sur volume/openInterest, et éventuellement bid/ask.

    option_type: 'call' | 'put' | None (les deux)
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    chain = tk.option_chain(expiry)

    frames = []
    if option_type in (None, "call"):
        calls = chain.calls.copy()
        calls["type"] = "call"
        frames.append(calls)
    if option_type in (None, "put"):
        puts = chain.puts.copy()
        puts["type"] = "put"
        frames.append(puts)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, axis=0, ignore_index=True)
    # Colonnes attendues Yahoo: contractSymbol, lastTradeDate, strike, lastPrice, bid, ask, change, percentChange,
    # volume, openInterest, impliedVolatility, inTheMoney, contractSize, currency

    df_out = pd.DataFrame()
    df_out["ticker"] = ticker
    df_out["expiry"] = expiry
    df_out["type"] = df["type"].astype(str)

    df_out["K"] = df["strike"].astype(float)
    df_out["last"] = df.get("lastPrice", np.nan)
    df_out["bid"] = df.get("bid", np.nan)
    df_out["ask"] = df.get("ask", np.nan)
    df_out["volume"] = df.get("volume", np.nan)
    df_out["openInterest"] = df.get("openInterest", np.nan)
    df_out["impliedVolatility"] = df.get("impliedVolatility", np.nan)

    # mid
    df_out["mid"] = [
        mid_price(b, a, l) for b, a, l in zip(df_out["bid"], df_out["ask"], df_out["last"])
    ]

    # maturité en années
    T = year_fraction(expiry)
    df_out["T"] = float(T)

    # nettoyage / filtres
    df_out = df_out.dropna(subset=["K"]).copy()

    if require_bid_ask:
        df_out = df_out[
            (df_out["bid"].astype(float) > 0) &
            (df_out["ask"].astype(float) > 0) &
            (df_out["ask"].astype(float) >= df_out["bid"].astype(float))
        ].copy()

    if min_volume > 0:
        df_out = df_out[df_out["volume"].fillna(0).astype(float) >= float(min_volume)].copy()

    if min_open_interest > 0:
        df_out = df_out[df_out["openInterest"].fillna(0).astype(float) >= float(min_open_interest)].copy()

    # mid non nul
    df_out = df_out[df_out["mid"].notna()].copy()
    df_out = df_out[df_out["mid"].astype(float) > 0].copy()

    df_out = df_out.sort_values(["type", "K"]).reset_index(drop=True)
    return df_out


def get_spot_price(ticker: str) -> Optional[float]:
    """
    Spot via yfinance : dernier close.
    """
    import yfinance as yf

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d")
    if hist is None or hist.empty:
        return None
    s = hist["Close"].dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])
