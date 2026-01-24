from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Dict
import numpy as np


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class VanillaOptionParams:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = "call"
    n_steps: int = 252
    n_paths: int = 50_000
    antithetic: bool = True
    seed: Optional[int] = 42


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

    def simulate_terminal(
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
        Simule uniquement S_T.
        Retour: array shape (n_paths_eff,)
        """
        if S0 <= 0 or T <= 0 or n_steps <= 0 or n_paths <= 0 or sigma < 0:
            raise ValueError("Paramètres invalides pour simulate_terminal.")

        dt = T / n_steps
        drift = (r - 0.5 * sigma**2) * dt
        vol = sigma * np.sqrt(dt)

        rng = np.random.default_rng(seed)

        if antithetic:
            half = max(n_paths // 2, 1)
            Z = rng.standard_normal(size=(half, n_steps))
            Z_full = np.vstack([Z, -Z])
        else:
            Z_full = rng.standard_normal(size=(n_paths, n_steps))

        # somme des incréments
        log_ST = np.sum(drift + vol * Z_full, axis=1)
        ST = S0 * np.exp(log_ST)
        return ST


class VanillaMCPricer:
    """
    Pricing Monte Carlo pour options vanilles européennes (call/put).
    """

    def __init__(self, simulator: Optional[GBMSimulator] = None):
        self.simulator = simulator if simulator is not None else GBMSimulator()

    @staticmethod
    def payoff(ST: np.ndarray, K: float, option_type: OptionType) -> np.ndarray:
        ST = np.asarray(ST)
        if option_type == "call":
            return np.maximum(ST - K, 0.0)
        else:
            return np.maximum(K - ST, 0.0)

    def price(self, p: VanillaOptionParams) -> MCResult:
        # T=0: payoff immédiat
        if p.T <= 0:
            payoff0 = max(p.S - p.K, 0.0) if p.option_type == "call" else max(p.K - p.S, 0.0)
            return MCResult(
                price=float(payoff0),
                stderr=0.0,
                ci_low=float(payoff0),
                ci_high=float(payoff0),
                details={"n_paths_eff": 0.0, "n_steps": float(p.n_steps), "antithetic": float(1.0 if p.antithetic else 0.0)},
            )

        ST = self.simulator.simulate_terminal(
            S0=p.S,
            T=p.T,
            r=p.r,
            sigma=p.sigma,
            n_steps=p.n_steps,
            n_paths=p.n_paths,
            antithetic=p.antithetic,
            seed=p.seed,
        )

        payoffs = self.payoff(ST, p.K, p.option_type)
        disc = np.exp(-p.r * p.T)
        discounted = disc * payoffs

        price = float(np.mean(discounted))
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
        p: VanillaOptionParams,
        eps_S: float = 0.1,
        eps_sigma: float = 1e-3,
        eps_r: float = 1e-4,
        eps_T: float = 1e-3,
    ) -> dict:
        """
        Greeks par différences finies centrales.
        Pour réduire le bruit MC, on garde le même seed entre les perturbations.
        """
        base = self.price(p).price

        # Delta
        p_up = VanillaOptionParams(**{**p.__dict__, "S": p.S + eps_S})
        p_dn = VanillaOptionParams(**{**p.__dict__, "S": max(p.S - eps_S, 1e-8)})
        delta = (self.price(p_up).price - self.price(p_dn).price) / (2.0 * eps_S)

        # Vega
        p_up = VanillaOptionParams(**{**p.__dict__, "sigma": p.sigma + eps_sigma})
        p_dn = VanillaOptionParams(**{**p.__dict__, "sigma": max(p.sigma - eps_sigma, 1e-12)})
        vega = (self.price(p_up).price - self.price(p_dn).price) / (2.0 * eps_sigma)

        # Rho
        p_up = VanillaOptionParams(**{**p.__dict__, "r": p.r + eps_r})
        p_dn = VanillaOptionParams(**{**p.__dict__, "r": p.r - eps_r})
        rho = (self.price(p_up).price - self.price(p_dn).price) / (2.0 * eps_r)

        # Theta : convention ici dV/dT (pas "time decay" négatif)
        eps_T_eff = min(eps_T, 0.1 * p.T) if p.T > 0 else eps_T
        p_dnT = VanillaOptionParams(**{**p.__dict__, "T": max(p.T - eps_T_eff, 1e-8)})
        theta = (self.price(p_dnT).price - base) / eps_T_eff

        return {
            "price": float(base),
            "delta": float(delta),
            "vega": float(vega),
            "rho": float(rho),
            "theta": float(theta),
        }
