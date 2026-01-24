from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict

import numpy as np


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class AsianOptionParams:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = "call"
    n_steps: int = 252          # nombre de pas de temps (observations)
    n_paths: int = 50_000       # nb trajectoires Monte Carlo
    antithetic: bool = True     # variance reduction simple
    seed: Optional[int] = 42    # reproductibilité
    include_S0_in_average: bool = True  # moyenne inclut S0 ou seulement S_t (t>0)


@dataclass(frozen=True)
class MCResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    details: Dict[str, float]


class GBMSimulator:
    """
    Simulateur GBM sous mesure risque-neutre:
    dS_t = r S_t dt + sigma S_t dW_t
    """

    def simulate_paths(
        self,
        S0: float,
        T: float,
        r: float,
        sigma: float,
        n_steps: int,
        n_paths: int,
        antithetic: bool = True,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Retourne paths de shape (n_paths_eff, n_steps+1), incluant S0 en colonne 0.
        Si antithetic=True, n_paths_eff sera pair et ~2*(n_paths//2).
        """
        if T <= 0 or n_steps <= 0 or n_paths <= 0 or S0 <= 0 or sigma < 0:
            raise ValueError("Paramètres invalides pour la simulation.")

        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        rng = np.random.default_rng(seed)

        if antithetic:
            half = n_paths // 2
            if half == 0:
                half = 1
            Z = rng.standard_normal(size=(half, n_steps))
            Z_full = np.vstack([Z, -Z])
        else:
            Z_full = rng.standard_normal(size=(n_paths, n_steps))

        n_paths_eff = Z_full.shape[0]

        # log paths
        increments = drift + vol * Z_full
        log_S = np.cumsum(increments, axis=1)
        log_S = np.hstack([np.zeros((n_paths_eff, 1)), log_S])  # include t=0
        paths = S0 * np.exp(log_S)
        return paths


class AsianArithmeticAveragePricer:
    """
    Pricing Monte Carlo pour options asiatiques average-price (moyenne arithmétique).
    """

    def __init__(self, simulator: Optional[GBMSimulator] = None):
        self.simulator = simulator if simulator is not None else GBMSimulator()

    @staticmethod
    def payoff_from_paths(paths: np.ndarray, K: float, option_type: OptionType, include_S0: bool) -> np.ndarray:
        """
        paths: (n_paths, n_steps+1)
        """
        if include_S0:
            avg = paths.mean(axis=1)
        else:
            avg = paths[:, 1:].mean(axis=1)

        if option_type == "call":
            return np.maximum(avg - K, 0.0)
        else:
            return np.maximum(K - avg, 0.0)

    def price(self, p: AsianOptionParams) -> MCResult:
        """
        Retourne prix + erreur standard + intervalle de confiance 95%.
        """
        # Cas T=0 : payoff direct basé sur moyenne (ici ~S0) => simple
        if p.T <= 0:
            avg = p.S if p.include_S0_in_average else p.S
            payoff0 = max(avg - p.K, 0.0) if p.option_type == "call" else max(p.K - avg, 0.0)
            return MCResult(price=float(payoff0), stderr=0.0, ci_low=float(payoff0), ci_high=float(payoff0), details={"n_paths": 0})

        paths = self.simulator.simulate_paths(
            S0=p.S,
            T=p.T,
            r=p.r,
            sigma=p.sigma,
            n_steps=p.n_steps,
            n_paths=p.n_paths,
            antithetic=p.antithetic,
            seed=p.seed,
        )

        payoffs = self.payoff_from_paths(paths, p.K, p.option_type, p.include_S0_in_average)
        disc = np.exp(-p.r * p.T)
        discounted = disc * payoffs

        price = float(np.mean(discounted))
        # stderr = sd / sqrt(n)
        sd = float(np.std(discounted, ddof=1)) if discounted.size > 1 else 0.0
        stderr = float(sd / np.sqrt(discounted.size)) if discounted.size > 0 else 0.0
        ci_low = float(price - 1.96 * stderr)
        ci_high = float(price + 1.96 * stderr)

        return MCResult(
            price=price,
            stderr=stderr,
            ci_low=ci_low,
            ci_high=ci_high,
            details={
                "n_paths_eff": float(discounted.size),
                "n_steps": float(p.n_steps),
                "antithetic": float(1.0 if p.antithetic else 0.0),
            },
        )

    def greeks_finite_diff(
        self,
        p: AsianOptionParams,
        eps_S: float = 0.1,
        eps_sigma: float = 1e-3,
        eps_r: float = 1e-4,
    ) -> dict:
        """
        Greeks par différences finies centrales.
        Important: Pour réduire le bruit MC, on utilise le même seed pour les perturbations.
        """
        base = self.price(p).price

        # Delta ~ dV/dS
        p_up = AsianOptionParams(**{**p.__dict__, "S": p.S + eps_S})
        p_dn = AsianOptionParams(**{**p.__dict__, "S": max(p.S - eps_S, 1e-8)})
        delta = (self.price(p_up).price - self.price(p_dn).price) / (2.0 * eps_S)

        # Vega ~ dV/dsigma
        p_up = AsianOptionParams(**{**p.__dict__, "sigma": p.sigma + eps_sigma})
        p_dn = AsianOptionParams(**{**p.__dict__, "sigma": max(p.sigma - eps_sigma, 1e-12)})
        vega = (self.price(p_up).price - self.price(p_dn).price) / (2.0 * eps_sigma)

        # Rho ~ dV/dr
        p_up = AsianOptionParams(**{**p.__dict__, "r": p.r + eps_r})
        p_dn = AsianOptionParams(**{**p.__dict__, "r": p.r - eps_r})
        rho = (self.price(p_up).price - self.price(p_dn).price) / (2.0 * eps_r)

        # Theta ~ dV/dT (approx backward difference pour T)
        eps_T = min(1e-3, 0.1 * p.T) if p.T > 0 else 1e-3
        p_t_dn = AsianOptionParams(**{**p.__dict__, "T": max(p.T - eps_T, 1e-8)})
        theta = (self.price(p_t_dn).price - base) / eps_T  # dV/dT approx (negative of time decay? depends convention)

        return {
            "price": float(base),
            "delta": float(delta),
            "vega": float(vega),
            "rho": float(rho),
            "theta": float(theta),
        }
