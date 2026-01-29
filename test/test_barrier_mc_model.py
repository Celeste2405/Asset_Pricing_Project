import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.barrier_mc_model import BarrierParams, BarrierMCPricer


def test_barrier_high_is_like_vanilla_call():
    pricer = BarrierMCPricer()
    p = BarrierParams(
        S=100, K=100, T=1.0, r=0.02, sigma=0.2,
        option_type="call",
        barrier_type="up-and-out",
        H=1e9, rebate=0.0,
        n_steps=252, n_paths=40_000, antithetic=True, seed=1
    )
    res = pricer.price(p)
    # if barrier is astronomically high, knockout rate should be ~0
    assert res.details["knockout_rate"] < 1e-4
    assert np.isfinite(res.price) and res.price > 0
    print(" High barrier behaves like vanilla | price:", res.price)


def test_barrier_low_knocks_out_almost_surely():
    pricer = BarrierMCPricer()
    p = BarrierParams(
        S=100, K=100, T=1.0, r=0.02, sigma=0.2,
        option_type="call",
        barrier_type="up-and-out",
        H=1.0, rebate=5.0,   # KO almost certain immediately
        n_steps=252, n_paths=40_000, antithetic=True, seed=2
    )
    res = pricer.price(p)
    assert res.details["knockout_rate"] > 0.999
    # price should be close to discounted rebate
    target = 5.0 * np.exp(-0.02 * 1.0)
    assert abs(res.price - target) < 0.3
    print(" Low barrier KO almost sure | price:", res.price, "| target:", target)


if __name__ == "__main__":
    test_barrier_high_is_like_vanilla_call()
    test_barrier_low_knocks_out_almost_surely()
