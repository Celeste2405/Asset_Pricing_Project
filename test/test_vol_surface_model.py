import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from model.calibration_model import OptionParams, BlackScholesModel
from model.vol_surface_model import VolSurfaceBuilder, VolSurfaceConfig


def test_constant_sigma_surface_recovers_sigma():
    """
    On génère des prix marché avec un sigma constant, puis on vérifie
    que la calibration retrouve ~sigma sur tout le grid.
    """
    S = 100.0
    r = 0.02
    true_sigma = 0.25

    Ks = np.array([80, 90, 100, 110, 120], dtype=float)
    Ts = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    bs = BlackScholesModel()
    rows = []
    for T in Ts:
        for K in Ks:
            p = OptionParams(S=S, K=K, T=T, r=r, option_type="call")
            price = bs.price(p, true_sigma)
            rows.append({"K": K, "T": T, "price_mkt": float(price), "type": "call"})
    df_opts = pd.DataFrame(rows)

    builder = VolSurfaceBuilder(config=VolSurfaceConfig(method="brent", sigma_min=1e-6, sigma_max=3.0))
    df_iv = builder.compute_iv_table(df_opts, S=S, r=r)

    # Vérif numérique : moyenne des erreurs sur sigma_imp
    err = np.nanmean(np.abs(df_iv["sigma_imp"].to_numpy() - true_sigma))
    assert err < 1e-6, f"Erreur moyenne trop grande: {err}"
    print("Surface sigma constant : sigma_imp retrouvé")

    surf = builder.pivot_surface(df_iv)
    assert surf.shape == (len(Ts), len(Ks))
    print("Pivot surface (T x K) OK")


def test_nonconstant_sigma_surface_sanity():
    """
    Cas plus réaliste : sigma dépend de K et T.
    On ne demande pas un fit parfait (BS ne produit pas vraiment un smile),
    mais on vérifie que:
      - sigma_imp est finie (pas tout NaN),
      - l'interpolation fonctionne.
    """
    S = 100.0
    r = 0.01

    Ks = np.array([80, 90, 100, 110, 120], dtype=float)
    Ts = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    bs = BlackScholesModel()
    rows = []
    for T in Ts:
        for K in Ks:
            # sigma "vrai" stylisé (smile + term structure)
            true_sigma = 0.20 + 0.15 * ((K - S) / S) ** 2 + 0.03 * np.sqrt(T)
            p = OptionParams(S=S, K=K, T=T, r=r, option_type="call")
            price = bs.price(p, true_sigma)
            rows.append({"K": K, "T": T, "price_mkt": float(price), "type": "call"})
    df_opts = pd.DataFrame(rows)

    builder = VolSurfaceBuilder(config=VolSurfaceConfig(method="brent", sigma_min=1e-6, sigma_max=3.0))
    df_iv = builder.compute_iv_table(df_opts, S=S, r=r)

    assert df_iv["sigma_imp"].notna().any(), "Tout est NaN -> calibration ratée"
    print(" Surface non-constante : sigma_imp calculée (au moins partiellement)")

    # Interpolation sur grille plus fine
    K_grid = np.linspace(Ks.min(), Ks.max(), 25)
    T_grid = np.linspace(Ts.min(), Ts.max(), 25)
    KK, TT, SIG = builder.interpolate_surface(df_iv, K_grid=K_grid, T_grid=T_grid, method="linear")

    assert KK.shape == TT.shape == SIG.shape
    print(" Interpolation OK")


if __name__ == "__main__":
    test_constant_sigma_surface_recovers_sigma()
    test_nonconstant_sigma_surface_sanity()
