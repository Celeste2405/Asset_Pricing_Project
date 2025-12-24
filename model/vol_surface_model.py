from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import griddata

from model.calibration_model import OptionParams, BlackScholesModel, ImpliedVolCalibrator

CalibMethod = Literal["brent", "newton"]


@dataclass(frozen=True)
class VolSurfaceConfig:
    """
    Configuration pour le calcul de vol implicite.
    """
    method: CalibMethod = "brent"
    sigma0: float = 0.2          # utilisé pour Newton
    sigma_min: float = 1e-6      # utilisé pour Brent
    sigma_max: float = 5.0       # utilisé pour Brent
    max_iter: int = 50
    tol: float = 1e-8


class VolSurfaceBuilder:
    """
    Construit une surface de volatilité implicite sigma_imp(K, T)
    à partir d'un tableau d'options.

    Entrée attendue (DataFrame):
    - K (strike) : float
    - T (maturité en années) : float
    - price_mkt : float
    - type : 'call' ou 'put' (optionnel -> défaut configuré côté app)
    """

    def __init__(
        self,
        model: Optional[BlackScholesModel] = None,
        calibrator: Optional[ImpliedVolCalibrator] = None,
        config: Optional[VolSurfaceConfig] = None,
    ):
        self.model = model if model is not None else BlackScholesModel()
        self.calibrator = calibrator if calibrator is not None else ImpliedVolCalibrator(self.model)
        self.config = config if config is not None else VolSurfaceConfig()

    @staticmethod
    def validate_options_df(df: pd.DataFrame) -> None:
        required = {"K", "T", "price_mkt"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Colonnes manquantes dans df options: {missing}. Requis: {required}")
        # types
        if "type" in df.columns:
            ok = df["type"].astype(str).str.lower().isin(["call", "put"])
            if not ok.all():
                bad = df.loc[~ok, "type"].unique()
                raise ValueError(f"Valeurs invalides dans colonne 'type': {bad}. Attendu 'call'/'put'.")

    def compute_iv_table(self, df_options: pd.DataFrame, S: float, r: float, default_type: str = "call") -> pd.DataFrame:
        """
        Retourne df avec une colonne sigma_imp calculée.
        """
        df = df_options.copy()
        self.validate_options_df(df)

        if "type" not in df.columns:
            df["type"] = default_type

        sigmas = []
        for _, row in df.iterrows():
            K = float(row["K"])
            T = float(row["T"])
            price_mkt = float(row["price_mkt"])
            opt_type = str(row["type"]).lower()
            opt_type = "call" if "call" in opt_type else "put"

            p = OptionParams(S=S, K=K, T=T, r=r, option_type=opt_type)

            sigma_imp = self.calibrator.implied_vol(
                market_price=price_mkt,
                p=p,
                method=self.config.method,
                sigma0=self.config.sigma0,
                sigma_min=self.config.sigma_min,
                sigma_max=self.config.sigma_max,
                max_iter=self.config.max_iter,
                tol=self.config.tol,
            )
            sigmas.append(sigma_imp)

        df["sigma_imp"] = sigmas
        return df

    @staticmethod
    def pivot_surface(df_iv: pd.DataFrame) -> pd.DataFrame:
        """
        Convertit la table IV en matrice (index=T, colonnes=K).
        """
        if "sigma_imp" not in df_iv.columns:
            raise ValueError("df_iv doit contenir une colonne 'sigma_imp'.")
        surf = (
            df_iv.pivot_table(index="T", columns="K", values="sigma_imp", aggfunc="mean")
            .sort_index()
            .sort_index(axis=1)
        )
        return surf

    @staticmethod
    def interpolate_surface(
        df_iv: pd.DataFrame,
        K_grid: np.ndarray,
        T_grid: np.ndarray,
        method: Literal["linear", "nearest", "cubic"] = "linear",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolation (optionnelle) sur une grille régulière (K_grid x T_grid).
        Retourne (KK, TT, SIG) où SIG est la surface interpolée.

        Note: griddata peut produire des NaN hors convex hull -> normal.
        """
        pts = df_iv[["K", "T"]].to_numpy(dtype=float)
        vals = df_iv["sigma_imp"].to_numpy(dtype=float)

        KK, TT = np.meshgrid(K_grid, T_grid)
        SIG = griddata(points=pts, values=vals, xi=(KK, TT), method=method)
        return KK, TT, SIG
