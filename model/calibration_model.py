import math
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]
CalibMethod = Literal["brent", "newton"]


@dataclass(frozen=True)
class OptionParams:
    S: float
    K: float
    T: float
    r: float
    option_type: OptionType = "call"


class BlackScholesModel:
    """
    Modèle Black-Scholes pour options européennes (call/put).
    Contient pricing + vega (utile pour Newton).
    """

    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    @staticmethod
    def _d2(d1: float, T: float, sigma: float) -> float:
        return d1 - sigma * math.sqrt(T)

    def price(self, p: OptionParams, sigma: float) -> float:
        if p.T <= 0 or sigma <= 0:
            if p.option_type == "call":
                return max(p.S - p.K, 0.0)
            return max(p.K - p.S, 0.0)

        d1 = self._d1(p.S, p.K, p.T, p.r, sigma)
        d2 = self._d2(d1, p.T, sigma)

        if p.option_type == "call":
            return p.S * norm.cdf(d1) - p.K * math.exp(-p.r * p.T) * norm.cdf(d2)
        else:
            return p.K * math.exp(-p.r * p.T) * norm.cdf(-d2) - p.S * norm.cdf(-d1)

    def vega(self, p: OptionParams, sigma: float) -> float:
        """
        Vega = dérivée du prix par rapport à sigma.
        (ici exprimée par unité de sigma, pas en %)
        """
        if p.T <= 0 or sigma <= 0:
            return 0.0
        d1 = self._d1(p.S, p.K, p.T, p.r, sigma)
        return p.S * norm.pdf(d1) * math.sqrt(p.T)


class ImpliedVolCalibrator:
    """
    Calibrage de volatilité implicite sigma_imp (aka omega_imp).
    Utilise un modèle (BS) et une méthode (brent/newton).
    """

    def __init__(self, model: Optional[BlackScholesModel] = None):
        self.model = model if model is not None else BlackScholesModel()

    def implied_vol(
        self,
        market_price: float,
        p: OptionParams,
        method: CalibMethod = "brent",
        sigma0: float = 0.2,
        sigma_min: float = 1e-6,
        sigma_max: float = 5.0,
        max_iter: int = 50,
        tol: float = 1e-8,
    ) -> float:
        if market_price < 0:
            return float("nan")

        if method == "newton":
            return self._newton(
                market_price=market_price, p=p, sigma0=sigma0, max_iter=max_iter, tol=tol
            )
        return self._brent(
            market_price=market_price, p=p, sigma_min=sigma_min, sigma_max=sigma_max
        )

    def _brent(self, market_price: float, p: OptionParams, sigma_min: float, sigma_max: float) -> float:
        def f(sig: float) -> float:
            return self.model.price(p, sig) - market_price

        try:
            return float(brentq(f, sigma_min, sigma_max))
        except ValueError:
            return float("nan")

    def _newton(self, market_price: float, p: OptionParams, sigma0: float, max_iter: int, tol: float) -> float:
        sigma = float(max(sigma0, 1e-8))

        for _ in range(max_iter):
            price = self.model.price(p, sigma)
            v = self.model.vega(p, sigma)
            if v < 1e-12:
                return float("nan")

            sigma_new = sigma - (price - market_price) / v
            sigma_new = float(max(sigma_new, 1e-12))

            if abs(sigma_new - sigma) < tol:
                return sigma_new
            sigma = sigma_new

        return float(sigma)
