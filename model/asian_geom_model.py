from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Dict, Any
import numpy as np
from scipy.stats import norm


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class AsianGeomParams:
    S: float
    K: float
    T: float
    r: float
    sigma: float
    option_type: OptionType = "call"

    n_steps: int = 252     # fixings (discret)
    n_paths: int = 50_000  # MC
    antithetic: bool = True
    seed: int = 42


@dataclass(frozen=True)
class PriceResult:
    price: float
    stderr: float
    ci_low: float
    ci_high: float
    details: Dict[str, Any]


def _geom_asian_equiv_params(S: float, T: float, r: float, sigma: float, n_steps: int) -> tuple[float, float]:
    """
    Paramètres équivalents pour l'average géométrique discret (fixings equally spaced).
    On approxime la distribution de ln(G) (G = moyenne géométrique) comme normale :
      ln G ~ N(m, v)

    Résultat : (mu_G, sigma_G) à utiliser dans une formule type BS sur "G".

    Formules classiques (Kemna & Vorst discret):
      sigma_G^2 = sigma^2 * ( (n+1)(2n+1) / (6 n^2) )
      mu_G = 0.5*(r - 0.5*sigma^2) * ( (n+1)/n ) + 0.5*sigma_G^2

    Ici on exprime le pricing en utilisant :
      G0 = S * exp( (r - 0.5*sigma^2)*A + 0.5*sigma^2*B )
    avec A, B en fonction de n.
    """
    n = int(n_steps)
    if n <= 1:
        n = 2

    # Variance effective (discret)
    var_factor = (n + 1) * (2 * n + 1) / (6 * n * n)
    sigma_G = sigma * np.sqrt(var_factor)

    # Drift effectif sur ln(G)
    # (n+1)/(2n) correspond à la moyenne des temps d'observation t_i / T
    A = (n + 1) / (2 * n) * T
    # Ajustement variance
    # ln(G) variance = sigma^2 * var_factor * T
    # mean ln(G) = ln(S) + (r - 0.5*sigma^2)*A + 0.5*sigma^2*var_factor*T? (cf. Kemna-Vorst discret)
    # On regroupe avec sigma_G^2 * T
    return A, sigma_G


def asian_geom_closed_form(S: float, K: float, T: float, r: float, sigma: float, option_type: OptionType, n_steps: int = 252) -> float:
    """
    Prix d'une option asiatique géométrique discrète (moyenne géométrique de fixings equally spaced).
    Call/Put sur G = (prod S_ti)^(1/n)
      payoff = max(G - K, 0) ou max(K - G, 0)

    Utilise l'approx lognormale de G (Kemna & Vorst / variantes discrètes).
    """
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        # payoff "instantané" approximatif
        intrinsic = max(S - K, 0.0) if option_type == "call" else max(K - S, 0.0)
        return float(intrinsic)

    A, sigma_G = _geom_asian_equiv_params(S, T, r, sigma, n_steps)

    # "Spot" équivalent sur G : E[G] sous Q sous forme exponentielle
    # On utilise approximation standard :
    # ln(G0) = ln(S) + (r - 0.5*sigma^2)*A + 0.5*(sigma_G^2)*T
    # (cohérent avec lognormal)
    lnG0 = np.log(S) + (r - 0.5 * sigma * sigma) * A + 0.5 * (sigma_G * sigma_G) * T
    G0 = np.exp(lnG0)

    # Black-like pricing on underlying G
    sigT = sigma_G * np.sqrt(T)
    d1 = (np.log(G0 / K) + (r + 0.5 * sigma_G * sigma_G) * T) / sigT
    d2 = d1 - sigT

    if option_type == "call":
        price = np.exp(-r * T) * (G0 * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - G0 * norm.cdf(-d1))

    return float(price)


class AsianGeomMCPricer:
    def simulate_paths(self, p: AsianGeomParams) -> np.ndarray:
        """
        Simule S_t sous GBM discret (Euler exact lognormal) aux dates equally spaced.
        Renvoie un array shape (n_paths, n_steps+1)
        """
        S0 = float(p.S)
        T = float(p.T)
        r = float(p.r)
        sig = float(p.sigma)
        n_steps = int(p.n_steps)
        n_paths = int(p.n_paths)

        dt = T / n_steps
        rng = np.random.default_rng(int(p.seed))

        # Z ~ N(0,1)
        Z = rng.standard_normal(size=(n_paths, n_steps))
        if p.antithetic:
            Z = np.vstack([Z, -Z])
        n_eff = Z.shape[0]

        increments = (r - 0.5 * sig * sig) * dt + sig * np.sqrt(dt) * Z
        logS = np.log(S0) + np.cumsum(increments, axis=1)
        S_paths = np.concatenate([np.full((n_eff, 1), S0), np.exp(logS)], axis=1)
        return S_paths

    @staticmethod
    def geometric_average(paths: np.ndarray) -> np.ndarray:
        """
        Moyenne géométrique sur les fixings (exclut t=0 pour être "fixings" sur 1..n_steps).
        """
        fix = paths[:, 1:]  # exclude S0
        return np.exp(np.mean(np.log(fix), axis=1))

    def price(self, p: AsianGeomParams) -> PriceResult:
        paths = self.simulate_paths(p)
        G = self.geometric_average(paths)

        if p.option_type == "call":
            payoff = np.maximum(G - p.K, 0.0)
        else:
            payoff = np.maximum(p.K - G, 0.0)

        disc = np.exp(-p.r * p.T)
        vals = disc * payoff

        price = float(np.mean(vals))
        stderr = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        ci_low = float(price - 1.96 * stderr)
        ci_high = float(price + 1.96 * stderr)

        return PriceResult(
            price=price,
            stderr=stderr,
            ci_low=ci_low,
            ci_high=ci_high,
            details={"n_paths_eff": len(vals), "n_steps": int(p.n_steps), "antithetic": bool(p.antithetic)},
        )
