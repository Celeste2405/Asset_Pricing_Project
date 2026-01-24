import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
from scipy.stats import norm

from model.asian import AsianOptionParams, AsianArithmeticAveragePricer


# --- helper vanilla BS (pour test limite n_steps=1) ---
def bs_call_price(S, K, T, r, sigma):
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def test_price_non_negative():
    pricer = AsianArithmeticAveragePricer()
    p = AsianOptionParams(S=100, K=100, T=1.0, r=0.02, sigma=0.2, option_type="call", n_steps=50, n_paths=20_000, seed=123)
    res = pricer.price(p)
    assert res.price >= 0.0
    assert res.stderr >= 0.0
    print("✅ Prix >= 0 et stderr >= 0")


def test_one_observation_approx_vanilla_call():
    """
    Si on observe uniquement à maturité (n_steps=1) et on n'inclut pas S0 dans la moyenne,
    alors la moyenne ~ S_T => payoff ~ vanilla.
    Donc Asian ≈ vanilla BS (MC avec assez de paths).
    """
    pricer = AsianArithmeticAveragePricer()

    S, K, T, r, sigma = 100, 100, 1.0, 0.02, 0.25

    p_asian = AsianOptionParams(
        S=S, K=K, T=T, r=r, sigma=sigma,
        option_type="call",
        n_steps=1,                 # une seule observation (t=T)
        n_paths=80_000,
        antithetic=True,
        seed=42,
        include_S0_in_average=False
    )

    res = pricer.price(p_asian)
    bs = bs_call_price(S, K, T, r, sigma)

    # tolérance : MC, donc on permet une petite marge
    assert abs(res.price - bs) < 0.5, f"Asian {res.price} vs BS {bs}"
    print("✅ n_steps=1 => Asian approx Vanilla (tol ok)")


def test_greeks_finite_diff_runs():
    pricer = AsianArithmeticAveragePricer()
    p = AsianOptionParams(S=100, K=95, T=0.7, r=0.01, sigma=0.3, option_type="put", n_steps=60, n_paths=30_000, seed=7)
    g = pricer.greeks_finite_diff(p, eps_S=0.2, eps_sigma=2e-3)
    for k in ["price", "delta", "vega", "rho", "theta"]:
        assert np.isfinite(g[k])
    print("✅ Greeks finite diff calculés (valeurs finies)")


if __name__ == "__main__":
    test_price_non_negative()
    test_one_observation_approx_vanilla_call()
    test_greeks_finite_diff_runs()
