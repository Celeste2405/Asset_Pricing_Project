import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.dupire_model import DupireLocalVol, DupireConfig


def test_dupire_constant_vol_is_constantish():
    S = 100.0
    r = 0.02
    sigma_const = 0.25

    Ks = np.array([80, 90, 100, 110, 120], dtype=float)
    Ts = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    mat_iv = pd.DataFrame(
        sigma_const * np.ones((len(Ts), len(Ks))),
        index=Ts,
        columns=Ks
    )

    dup = DupireLocalVol()
    cfg = DupireConfig(r=r, denom_floor=1e-10, vol_floor=1e-6, vol_cap=5.0)

    _, mat_loc = dup.local_vol_from_iv_surface(S=S, r=r, mat_iv=mat_iv, cfg=cfg)

    # On ignore les bords (numériquement plus instables)
    core = mat_loc.iloc[1:-1, 1:-1].values
    core = core[np.isfinite(core)]

    assert core.size > 0
    err = np.abs(core - sigma_const)
    assert np.nanmedian(err) < 0.05  # tolérance large (finite diff)
    print("✅ Dupire constant vol -> local vol ~ constant | median abs err:", float(np.nanmedian(err)))


if __name__ == "__main__":
    test_dupire_constant_vol_is_constantish()
