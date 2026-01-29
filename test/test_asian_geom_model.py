import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.asian_geom_model import asian_geom_closed_form, AsianGeomParams, AsianGeomMCPricer


def test_geom_asian_mc_matches_closed_form_reasonably():
    S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.02, 0.2
    n_steps = 252

    cf = asian_geom_closed_form(S, K, T, r, sigma, "call", n_steps=n_steps)

    pricer = AsianGeomMCPricer()
    p = AsianGeomParams(
        S=S, K=K, T=T, r=r, sigma=sigma,
        option_type="call",
        n_steps=n_steps,
        n_paths=80_000,
        antithetic=True,
        seed=123
    )
    mc = pricer.price(p)

    # Doit être dans quelques std errors
    diff = abs(mc.price - cf)
    assert diff < 6.0 * mc.stderr
    print(" MC close to closed-form | cf:", cf, "| mc:", mc.price, "| stderr:", mc.stderr)


if __name__ == "__main__":
    test_geom_asian_mc_matches_closed_form_reasonably()
