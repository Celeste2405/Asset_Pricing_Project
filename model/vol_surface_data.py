from __future__ import annotations
import sys

from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.market_data import list_expiries, get_option_chain, get_spot_price


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class YahooSurfaceConfig:
    ticker: str
    option_type: OptionType = "call"

    # sélection expiries
    max_expiries: int = 6  # limiter pour éviter trop de requêtes

    # filtrage strikes autour du spot
    moneyness_low: float = 0.7   # K >= 0.7*S
    moneyness_high: float = 1.3  # K <= 1.3*S

    # filtres de liquidité
    require_bid_ask: bool = True
    min_open_interest: int = 0
    min_volume: int = 0


def build_yahoo_option_dataset(cfg: YahooSurfaceConfig) -> tuple[pd.DataFrame, float]:
    """
    Construit un DataFrame standardisé:
      columns: ['type','K','T','price_mkt','bid','ask','last','volume','openInterest','impliedVolatility','expiry','ticker']
    Retourne (df, spot)

    Remarque:
    - price_mkt = mid (bid/ask) sinon last
    - T = year fraction (ACT/365) déjà fourni par market_data.get_option_chain
    """
    spot = get_spot_price(cfg.ticker)
    if spot is None or spot <= 0:
        raise ValueError("Impossible de récupérer le spot (close) du ticker Yahoo.")

    expiries = list_expiries(cfg.ticker)
    if not expiries:
        raise ValueError("Aucune maturité trouvée pour ce ticker.")

    expiries = expiries[: max(1, int(cfg.max_expiries))]

    frames: list[pd.DataFrame] = []
    for exp in expiries:
        df_chain = get_option_chain(
            ticker=cfg.ticker,
            expiry=exp,
            option_type=cfg.option_type,
            min_volume=int(cfg.min_volume),
            min_open_interest=int(cfg.min_open_interest),
            require_bid_ask=bool(cfg.require_bid_ask),
        )
        if df_chain.empty:
            continue

        # filtre moneyness
        lowK = cfg.moneyness_low * spot
        highK = cfg.moneyness_high * spot
        df_chain = df_chain[(df_chain["K"].astype(float) >= lowK) & (df_chain["K"].astype(float) <= highK)].copy()
        if df_chain.empty:
            continue

        # standardisation
        out = pd.DataFrame()
        out["ticker"] = cfg.ticker
        out["expiry"] = exp
        out["type"] = df_chain["type"].astype(str)
        out["K"] = df_chain["K"].astype(float)
        out["T"] = df_chain["T"].astype(float)
        out["price_mkt"] = df_chain["mid"].astype(float)

        # colonnes utiles debug
        for col in ["bid", "ask", "last", "volume", "openInterest", "impliedVolatility"]:
            if col in df_chain.columns:
                out[col] = df_chain[col]
            else:
                out[col] = np.nan

        frames.append(out)

    if not frames:
        raise ValueError("Aucune option ne passe les filtres (essaie d’élargir moneyness ou diminuer filtres).")

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["K", "T", "price_mkt"]).copy()
    df = df[df["price_mkt"].astype(float) > 0].copy()

    # tri utile
    df = df.sort_values(["T", "K"]).reset_index(drop=True)
    return df, float(spot)
