import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
from model.calibration_model import OptionParams, BlackScholesModel, ImpliedVolCalibrator


def test_implied_vol_recovers_true_sigma_call():
    model = BlackScholesModel()
    calib = ImpliedVolCalibrator(model)

    p = OptionParams(S=100, K=100, T=1.0, r=0.02, option_type="call")
    true_sigma = 0.25

    market_price = model.price(p, true_sigma)

    sigma_brent = calib.implied_vol(market_price, p, method="brent")
    sigma_newton = calib.implied_vol(market_price, p, method="newton", sigma0=0.2)

    assert abs(sigma_brent - true_sigma) < 1e-6
    assert abs(sigma_newton - true_sigma) < 1e-6

    print("CALL: Brent & Newton retrouvent sigma vrai")


def test_implied_vol_price_check_put():
    model = BlackScholesModel()
    calib = ImpliedVolCalibrator(model)

    p = OptionParams(S=100, K=110, T=0.5, r=0.01, option_type="put")
    true_sigma = 0.30

    market_price = model.price(p, true_sigma)
    sigma = calib.implied_vol(market_price, p, method="brent")

    assert not np.isnan(sigma)
    price_check = model.price(p, sigma)

    assert abs(price_check - market_price) < 1e-6

    print("✅ PUT: prix BS(sigma_imp) ≈ prix marché")


if __name__ == "__main__":
    test_implied_vol_recovers_true_sigma_call()
    test_implied_vol_price_check_put()
