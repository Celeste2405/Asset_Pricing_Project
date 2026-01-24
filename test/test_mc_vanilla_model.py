import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
from scipy.stats import norm

from model.mc_vanilla_model import VanillaOptionParams, VanillaMCPricer


# --- Black-Scholes (référence pour tests) ---
def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def test_mc_price_non_negative():
    pricer = VanillaMCPricer()
    p = VanillaOptionParams(S=100, K=100, T=1.0, r=0.02, sigma=0.2, option_type="call", n_steps=252, n_paths=20_000, antithetic=True, seed=123)
    res = pricer.price(p)
    assert res.price >= 0.0
    assert res.stderr >= 0.0
    assert res.ci_low <= res.price <= res.ci_high
    print("✅ MC non-neg + IC cohérent")


def test_mc_converges_to_bs_call():
    pricer = VanillaMCPricer()

    S, K, T, r, sigma = 100, 100, 1.0, 0.02, 0.25
    bs = bs_call_price(S, K, T, r, sigma)

    p = VanillaOptionParams(
        S=S, K=K, T=T, r=r, sigma=sigma,
        option_type="call",
        n_steps=252,
        n_paths=120_000,
        antithetic=True,
        seed=42
    )
    res = pricer.price(p)

    # Tolérance MC : à ajuster selon machine/variance (ici assez large mais réaliste)
    assert abs(res.price - bs) < 0.5, f"MC {res.price} vs BS {bs}"
    print("✅ Convergence MC ~ BS (call)")


def test_put_call_parity_approx():
    """
    Put-call parity: C - P = S - K e^{-rT}
    MC => approximation.
    """
    pricer = VanillaMCPricer()
    S, K, T, r, sigma = 100, 105, 1.0, 0.03, 0.2

    p_call = VanillaOptionParams(S=S, K=K, T=T, r=r, sigma=sigma, option_type="call", n_steps=252, n_paths=120_000, antithetic=True, seed=7)
    p_put  = VanillaOptionParams(S=S, K=K, T=T, r=r, sigma=sigma, option_type="put",  n_steps=252, n_paths=120_000, antithetic=True, seed=7)

    C = pricer.price(p_call).price
    P = pricer.price(p_put).price

    rhs = S - K * np.exp(-r * T)
    lhs = C - P

    assert abs(lhs - rhs) < 0.8, f"lhs {lhs} rhs {rhs}"
    print("✅ Put-call parity approx (MC)")


def test_greeks_runs():
    pricer = VanillaMCPricer()
    p = VanillaOptionParams(S=100, K=95, T=0.7, r=0.01, sigma=0.3, option_type="put", n_steps=100, n_paths=60_000, antithetic=True, seed=11)
    g = pricer.greeks_finite_diff(p, eps_S=0.2, eps_sigma=2e-3, eps_r=1e-4, eps_T=1e-3)
    for k in ["price", "delta", "vega", "rho", "theta"]:
        assert np.isfinite(g[k])
    print("✅ Greeks finite diff OK")


if __name__ == "__main__":
    test_mc_price_non_negative()
    test_mc_converges_to_bs_call()
    test_put_call_parity_approx()
    test_greeks_runs()
