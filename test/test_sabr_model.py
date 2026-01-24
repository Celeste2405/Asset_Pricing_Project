import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.sabr_model import SABRParams, hagan_lognormal_iv, SABRCalibrator


def test_hagan_iv_finite():
    p = SABRParams(alpha=0.2, beta=0.5, rho=-0.2, nu=0.8)
    v = hagan_lognormal_iv(F=100, K=100, T=1.0, p=p)
    assert np.isfinite(v) and v > 0
    print("✅ Hagan ATM vol finite")


def test_calibration_runs_on_synthetic():
    # Synthetic "market" from known SABR params
    F = 100.0
    T = 1.0
    Ks = np.array([80, 90, 100, 110, 120], dtype=float)

    true = SABRParams(alpha=0.25, beta=0.5, rho=-0.3, nu=0.9)
    iv_mkt = np.array([hagan_lognormal_iv(F, k, T, true) for k in Ks], dtype=float)

    cal = SABRCalibrator(beta=0.5)
    est, loss = cal.calibrate(F=F, Ks=Ks, T=T, iv_mkt=iv_mkt, max_iter=150)

    assert np.isfinite(loss)
    assert est.alpha > 0 and est.nu > 0
    assert -1 < est.rho < 1
    print("✅ Calibration synthetic ok | loss:", loss)


if __name__ == "__main__":
    test_hagan_iv_finite()
    test_calibration_runs_on_synthetic()
