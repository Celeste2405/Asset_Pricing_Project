from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.calibration_model import OptionParams, BlackScholesModel




@dataclass(frozen=True)
class DupireConfig:
    r: float = 0.02
    # dérivées numériques
    t_eps: float = 1e-4   # pas pour d/dT
    k_eps: float = 1e-2   # pas pour d/dK (si grille non uniforme on utilise diff centrée de numpy)
    # stabilité
    denom_floor: float = 1e-12
    vol_floor: float = 1e-6
    vol_cap: float = 5.0


class DupireLocalVol:
    """
    Pipeline:
      1) Convertir une surface sigma(K,T) -> surface de prix de CALL C(K,T) via BS
      2) Calculer dC/dT et d²C/dK² par différences finies sur la grille (T,K)
      3) Appliquer Dupire: sigma_loc^2 = dT / (0.5*K^2*dKK)
    """

    def __init__(self):
        self.bs = BlackScholesModel()

    def call_price_surface_from_iv(
        self,
        S: float,
        r: float,
        mat_iv: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        mat_iv: DataFrame index=T, columns=K, values = sigma(K,T)
        Returns: mat_C with same shape (call prices)
        """
        Ts = np.asarray(mat_iv.index, dtype=float)
        Ks = np.asarray(mat_iv.columns, dtype=float)

        C = np.full((len(Ts), len(Ks)), np.nan, dtype=float)

        for i, T in enumerate(Ts):
            for j, K in enumerate(Ks):
                sigma = float(mat_iv.iloc[i, j])
                if not np.isfinite(sigma) or sigma <= 0 or T <= 0 or K <= 0:
                    continue
                p = OptionParams(S=float(S), K=float(K), T=float(T), r=float(r), option_type="call")
                C[i, j] = float(self.bs.price(p, sigma))

        return pd.DataFrame(C, index=mat_iv.index, columns=mat_iv.columns)

    @staticmethod
    def _d_dt(mat: np.ndarray, Ts: np.ndarray) -> np.ndarray:
        """
        d/dT using numpy gradient along axis 0, with non-uniform spacing Ts.
        """
        return np.gradient(mat, Ts, axis=0)

    @staticmethod
    def _d2_dk2(mat: np.ndarray, Ks: np.ndarray) -> np.ndarray:
        """
        d²/dK² using two gradients (central finite differences), non-uniform spacing Ks.
        """
        d_dk = np.gradient(mat, Ks, axis=1)
        d2_dk2 = np.gradient(d_dk, Ks, axis=1)
        return d2_dk2

    def local_vol_from_call_surface(
        self,
        mat_C: pd.DataFrame,
        cfg: DupireConfig,
    ) -> pd.DataFrame:
        """
        mat_C: DataFrame index=T, columns=K, values = Call prices C(K,T)
        Returns: mat_sigma_loc (T x K)
        """
        Ts = np.asarray(mat_C.index, dtype=float)
        Ks = np.asarray(mat_C.columns, dtype=float)

        C = mat_C.values.astype(float)
        dC_dT = self._d_dt(C, Ts)
        d2C_dK2 = self._d2_dk2(C, Ks)

        # Dupire formula
        KK = Ks.reshape(1, -1)  # broadcast
        denom = 0.5 * (KK ** 2) * d2C_dK2

        # guards
        denom = np.where(np.abs(denom) < cfg.denom_floor, np.nan, denom)
        sigma2 = dC_dT / denom

        # negative / nan guards
        sigma2 = np.where(np.isfinite(sigma2) & (sigma2 > 0), sigma2, np.nan)
        sigma = np.sqrt(sigma2)

        # clip
        sigma = np.where(np.isfinite(sigma), np.clip(sigma, cfg.vol_floor, cfg.vol_cap), np.nan)

        return pd.DataFrame(sigma, index=mat_C.index, columns=mat_C.columns)

    def local_vol_from_iv_surface(
        self,
        S: float,
        r: float,
        mat_iv: pd.DataFrame,
        cfg: DupireConfig,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Convenience:
          IV surface -> Call surface -> Local vol surface
        Returns (mat_C, mat_sigma_loc)
        """
        mat_C = self.call_price_surface_from_iv(S=float(S), r=float(r), mat_iv=mat_iv)
        mat_loc = self.local_vol_from_call_surface(mat_C=mat_C, cfg=cfg)
        return mat_C, mat_loc
