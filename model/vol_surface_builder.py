
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.calibration_model import OptionParams, BlackScholesModel, ImpliedVolCalibrator


Method = Literal["brent", "newton"]


@dataclass(frozen=True)
class SurfaceBuildConfig:
    r: float = 0.02
    option_type: Literal["call", "put"] = "call"

    method: Method = "brent"
    sigma0: float = 0.20
    sigma_min: float = 1e-6
    sigma_max: float = 5.0

    # nettoyage : on élimine les σ_imp absurdes
    vol_floor: float = 1e-6
    vol_cap: float = 5.0


class VolSurfaceBuilder:
    """
    Construit σ_imp(K,T) sur un dataset d'options:
      df doit contenir au minimum: ['K','T','price_mkt']
    """

    def __init__(self):
        self.bs = BlackScholesModel()
        self.cal = ImpliedVolCalibrator(self.bs)

    def build_surface(
        self,
        df_opts: pd.DataFrame,
        S: float,
        cfg: SurfaceBuildConfig,
    ) -> pd.DataFrame:
        """
        Retourne df avec colonne 'sigma_imp' en plus.
        """
        need = {"K", "T", "price_mkt"}
        missing = need - set(df_opts.columns)
        if missing:
            raise ValueError(f"df_opts missing columns: {missing}")

        rows = []
        for _, row in df_opts.iterrows():
            K = float(row["K"])
            T = float(row["T"])
            price_mkt = float(row["price_mkt"])

            p = OptionParams(S=float(S), K=K, T=T, r=float(cfg.r), option_type=str(cfg.option_type))
            sigma = self.cal.implied_vol(
                market_price=price_mkt,
                p=p,
                method=str(cfg.method),
                sigma0=float(cfg.sigma0),
                sigma_min=float(cfg.sigma_min),
                sigma_max=float(cfg.sigma_max),
            )

            rows.append(float(sigma) if sigma is not None else np.nan)

        out = df_opts.copy()
        out["S"] = float(S)
        out["r"] = float(cfg.r)
        out["sigma_imp"] = rows

        # nettoyage
        out["sigma_imp"] = pd.to_numeric(out["sigma_imp"], errors="coerce")
        out = out.dropna(subset=["sigma_imp"]).copy()
        out = out[(out["sigma_imp"] >= cfg.vol_floor) & (out["sigma_imp"] <= cfg.vol_cap)].copy()
        out = out.sort_values(["T", "K"]).reset_index(drop=True)
        return out

    @staticmethod
    def pivot_surface(df_surface: pd.DataFrame) -> pd.DataFrame:
        """
        Retourne une matrice (index=T, columns=K) avec σ_imp.
        """
        mat = df_surface.pivot_table(index="T", columns="K", values="sigma_imp", aggfunc="mean")
        mat = mat.sort_index().sort_index(axis=1)
        return mat
