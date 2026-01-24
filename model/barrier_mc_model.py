from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any
import numpy as np


OptionType = Literal["call", "put"]
BarrierType = Literal["up-and-out", "down-and-out"]


@dataclass(frozen=True)
class BarrierParams:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = "call"

    barrier_type: BarrierType = "up-and-out"
    H: float = 120.0         # barrier level
    rebate: float = 0.0      # paid if knocked out (here paid at maturity, simple)

    n_steps: int = 252
    n_paths: int = 80_000
    antithetic: bool = True
    seed: int = 42


@dataclass(frozen=True)
class PriceResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    details: Dict[str, Any]


class BarrierMCPricer:
    """
    Monte Carlo pricing for barrier "knock-out" options under GBM.
    Barrier monitoring is DISCRETE on the simulated grid (n_steps).
    """

    def simulate_paths(self, p: BarrierParams) -> np.ndarray:
        S0 = float(p.S)
        T = float(p.T)
        r = float(p.r)
        sig = float(p.sigma)
        n_steps = int(p.n_steps)
        n_paths = int(p.n_paths)

        if S0 <= 0 or T <= 0 or sig < 0:
            raise ValueError("Bad parameters: require S>0, T>0, sigma>=0.")

        dt = T / n_steps
        rng = np.random.default_rng(int(p.seed))

        Z = rng.standard_normal(size=(n_paths, n_steps))
        if p.antithetic:
            Z = np.vstack([Z, -Z])

        drift = (r - 0.5 * sig * sig) * dt
        diff = sig * np.sqrt(dt) * Z

        logS = np.log(S0) + np.cumsum(drift + diff, axis=1)
        paths = np.concatenate([np.full((Z.shape[0], 1), S0), np.exp(logS)], axis=1)
        return paths

    @staticmethod
    def _knocked_out(paths: np.ndarray, barrier_type: str, H: float) -> np.ndarray:
        """
        Returns boolean array shape (n_paths_eff,)
        """
        H = float(H)
        if barrier_type == "up-and-out":
            return np.any(paths >= H, axis=1)  # touched or exceeded
        elif barrier_type == "down-and-out":
            return np.any(paths <= H, axis=1)
        else:
            raise ValueError("barrier_type must be 'up-and-out' or 'down-and-out'")

    @staticmethod
    def _vanilla_payoff(ST: np.ndarray, K: float, option_type: str) -> np.ndarray:
        K = float(K)
        if option_type == "call":
            return np.maximum(ST - K, 0.0)
        elif option_type == "put":
            return np.maximum(K - ST, 0.0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

    def price(self, p: BarrierParams) -> PriceResult:
        # quick degenerate checks
        if p.T <= 0:
            intrinsic = max(p.S - p.K, 0.0) if p.option_type == "call" else max(p.K - p.S, 0.0)
            return PriceResult(price=float(intrinsic), stderr=0.0, ci_low=float(intrinsic), ci_high=float(intrinsic), details={"degenerate": True})

        paths = self.simulate_paths(p)
        ST = paths[:, -1]

        knocked = self._knocked_out(paths, p.barrier_type, p.H)

        vanilla = self._vanilla_payoff(ST, p.K, p.option_type)

        # knock-out: if knocked -> rebate, else vanilla payoff
        payoff = np.where(knocked, float(p.rebate), vanilla)

        disc = np.exp(-p.r * p.T)
        vals = disc * payoff

        price = float(np.mean(vals))
        stderr = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ci_low = float(price - 1.96 * stderr)
        ci_high = float(price + 1.96 * stderr)

        details = {
            "n_paths_eff": int(len(vals)),
            "n_steps": int(p.n_steps),
            "antithetic": bool(p.antithetic),
            "knockout_rate": float(np.mean(knocked)),
            "barrier_type": p.barrier_type,
            "H": float(p.H),
        }

        return PriceResult(price=price, stderr=stderr, ci_low=ci_low, ci_high=ci_high, details=details)
