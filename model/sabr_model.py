from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class SABRParams:
    alpha: float   # > 0
    beta: float    # in [0,1]
    rho: float     # in (-1,1)
    nu: float      # > 0


def hagan_lognormal_iv(F: float, K: float, T: float, p: SABRParams) -> float:
    """
    Hagan SABR lognormal implied vol approximation (Black vol), for F>0, K>0.
    """
    F = float(F)
    K = float(K)
    T = float(T)

    if F <= 0 or K <= 0 or T <= 0:
        return np.nan

    alpha, beta, rho, nu = p.alpha, p.beta, p.rho, p.nu
    if alpha <= 0 or nu <= 0 or not (0 <= beta <= 1) or not (-0.999 < rho < 0.999):
        return np.nan

    # Handle ATM separately for stability
    if abs(F - K) < 1e-12:
        FK = F
        one_minus_beta = 1.0 - beta
        term1 = alpha / (FK ** one_minus_beta)
        # correction terms
        c1 = (one_minus_beta**2 * alpha**2) / (24.0 * (FK ** (2.0 * one_minus_beta)))
        c2 = (rho * beta * nu * alpha) / (4.0 * (FK ** one_minus_beta))
        c3 = (2.0 - 3.0 * rho**2) * nu**2 / 24.0
        return float(term1 * (1.0 + (c1 + c2 + c3) * T))

    # general case
    one_minus_beta = 1.0 - beta
    FK = F * K
    z = (nu / alpha) * (FK ** (0.5 * one_minus_beta)) * np.log(F / K)
    # x(z)
    xz = np.log((np.sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))

    # prefactor
    denom = (FK ** (0.5 * one_minus_beta))
    A = alpha / denom

    # log(F/K) corrections
    logFK = np.log(F / K)
    B = 1.0 + (one_minus_beta**2 / 24.0) * (logFK**2) + (one_minus_beta**4 / 1920.0) * (logFK**4)

    # time correction
    c1 = (one_minus_beta**2 * alpha**2) / (24.0 * (FK ** (one_minus_beta)))
    c2 = (rho * beta * nu * alpha) / (4.0 * (FK ** (0.5 * one_minus_beta)))
    c3 = (2.0 - 3.0 * rho**2) * nu**2 / 24.0

    vol = (A / B) * (z / xz) * (1.0 + (c1 + c2 + c3) * T)
    return float(vol)


def sabr_iv_vector(F: float, Ks: np.ndarray, T: float, p: SABRParams) -> np.ndarray:
    Ks = np.asarray(Ks, dtype=float)
    return np.array([hagan_lognormal_iv(F, k, T, p) for k in Ks], dtype=float)


class SABRCalibrator:
    """
    Calibrate SABR params (alpha, rho, nu) for a given maturity, beta fixed.
    We minimize squared error between SABR IV and market IV.
    """

    def __init__(self, beta: float = 0.5):
        self.beta = float(beta)

    def _clip_params(self, x: np.ndarray) -> SABRParams:
        # x = [log_alpha, atanh_rho, log_nu]  -> unconstrained
        alpha = float(np.exp(x[0]))
        rho = float(np.tanh(x[1]))
        nu = float(np.exp(x[2]))
        return SABRParams(alpha=alpha, beta=self.beta, rho=rho, nu=nu)

    def objective(self, x: np.ndarray, F: float, Ks: np.ndarray, T: float, iv_mkt: np.ndarray) -> float:
        p = self._clip_params(x)
        iv_model = sabr_iv_vector(F, Ks, T, p)
        mask = np.isfinite(iv_model) & np.isfinite(iv_mkt)
        if mask.sum() < 3:
            return 1e6
        err = iv_model[mask] - iv_mkt[mask]
        return float(np.mean(err**2))

    def calibrate(
        self,
        F: float,
        Ks: np.ndarray,
        T: float,
        iv_mkt: np.ndarray,
        x0: Optional[np.ndarray] = None,
        max_iter: int = 200,
        lr: float = 0.05,
        tol: float = 1e-8,
    ) -> Tuple[SABRParams, float]:
        """
        Simple derivative-free "coordinate + random" search (no scipy optimize dependency).
        Returns (params, loss).

        Note: for better calibration you can swap to scipy.optimize later.
        """
        Ks = np.asarray(Ks, dtype=float)
        iv_mkt = np.asarray(iv_mkt, dtype=float)

        if x0 is None:
            # alpha ~ atm vol * F^(1-beta), rho ~ 0, nu ~ 0.5
            atm_idx = int(np.argmin(np.abs(Ks - F)))
            atm_iv = float(iv_mkt[atm_idx]) if np.isfinite(iv_mkt[atm_idx]) else float(np.nanmean(iv_mkt))
            atm_iv = max(atm_iv, 1e-4)
            alpha0 = atm_iv * (F ** (1.0 - self.beta))
            x = np.array([np.log(max(alpha0, 1e-6)), np.arctanh(0.0), np.log(0.5)], dtype=float)
        else:
            x = np.array(x0, dtype=float)

        best = self.objective(x, F, Ks, T, iv_mkt)

        rng = np.random.default_rng(42)
        step = np.array([0.2, 0.4, 0.2], dtype=float)

        for _ in range(max_iter):
            improved = False
            # coordinate moves
            for j in range(3):
                for sgn in (+1.0, -1.0):
                    x_try = x.copy()
                    x_try[j] += sgn * step[j] * lr
                    val = self.objective(x_try, F, Ks, T, iv_mkt)
                    if val + tol < best:
                        x, best = x_try, val
                        improved = True

            # small random perturbation if stuck
            if not improved:
                x_try = x + rng.normal(0.0, step * 0.15)
                val = self.objective(x_try, F, Ks, T, iv_mkt)
                if val + tol < best:
                    x, best = x_try, val
                    improved = True
                else:
                    step *= 0.98  # anneal

            if step.max() < 1e-3:
                break

        return self._clip_params(x), float(best)
