import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import yfinance as yf

# ---------------------------------------------------------------------
# Import projet (robuste quel que soit le dossier d'exécution)
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.vol_surface_model import VolSurfaceBuilder, VolSurfaceConfig
from model.calibration_model import BlackScholesModel, OptionParams


# ---------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------
st.set_page_config(page_title="Vol Surface", layout="wide")
st.title("3) Vol Surface : σ_imp(K, T)")
st.write(
    "Cette page construit une surface de volatilité implicite à partir d’un set de prix d’options.\n\n"
    "- **A) Simulées** : démo robuste (toujours dispo)\n"
    "- **B) Yahoo Finance** : données de marché (options chain via yfinance)\n"
)
st.info("⚠️ r est un **input utilisateur** (consigne prof).")


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def compute_T_years(expiry_str: str, today: date | None = None) -> float:
    """Convertit une date d'expiration 'YYYY-MM-DD' en maturité T (années)."""
    if today is None:
        today = date.today()
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").date()
    days = (exp - today).days
    return max(days / 365.0, 0.0)


def yahoo_extract_options(
    ticker: str,
    option_side: str,
    max_expiries: int,
    use_mid_only: bool,
    strike_min_mult: float,
    strike_max_mult: float,
    min_days_to_expiry: int,
) -> tuple[pd.DataFrame, float, dict]:
    """
    Extrait un tableau d'options depuis Yahoo via yfinance avec filtres qualité.
    Retourne (df_opts, S_spot, stats)
    df_opts colonnes: K, T, price_mkt, type
    """
    t = yf.Ticker(ticker)

    # Spot robuste
    try:
        hist = t.history(period="5d")
        if hist.empty:
            raise ValueError("Historique vide pour ce ticker.")
        S_spot = float(hist["Close"].dropna().iloc[-1])
    except Exception:
        info = t.info
        S_spot = float(info.get("regularMarketPrice", np.nan))

    if not np.isfinite(S_spot) or S_spot <= 0:
        raise ValueError("Impossible de récupérer un spot valide (S) sur Yahoo.")

    expiries = list(t.options or [])
    if len(expiries) == 0:
        raise ValueError("Aucune maturité d’options disponible sur Yahoo pour ce ticker.")
    expiries = expiries[:max_expiries]

    today = date.today()
    rows = []

    stats = {
        "expiries_total": len(expiries),
        "rows_seen": 0,
        "kept": 0,
        "dropped_bad_T": 0,
        "dropped_bad_strike": 0,
        "dropped_bad_quotes": 0,
        "dropped_bad_price": 0,
    }

    K_min = strike_min_mult * S_spot
    K_max = strike_max_mult * S_spot
    min_T = min_days_to_expiry / 365.0

    for exp in expiries:
        T = compute_T_years(exp, today=today)
        if T < min_T:
            stats["dropped_bad_T"] += 1
            continue

        chain = t.option_chain(exp)
        df_chain = chain.calls if option_side == "call" else chain.puts

        # Parcours lignes
        for i in range(len(df_chain)):
            stats["rows_seen"] += 1

            K = float(df_chain["strike"].iloc[i])
            if (K <= 0) or (K < K_min) or (K > K_max):
                stats["dropped_bad_strike"] += 1
                continue

            bid = float(df_chain["bid"].iloc[i]) if "bid" in df_chain.columns else np.nan
            ask = float(df_chain["ask"].iloc[i]) if "ask" in df_chain.columns else np.nan
            last = float(df_chain["lastPrice"].iloc[i]) if "lastPrice" in df_chain.columns else np.nan

            # Filtre liquidité / quotes
            if use_mid_only:
                if (not np.isfinite(bid)) or (not np.isfinite(ask)) or bid <= 0 or ask <= 0:
                    stats["dropped_bad_quotes"] += 1
                    continue
                price_mkt = 0.5 * (bid + ask)
            else:
                price_mkt = np.nan
                if np.isfinite(bid) and np.isfinite(ask) and bid > 0 and ask > 0:
                    price_mkt = 0.5 * (bid + ask)
                if (not np.isfinite(price_mkt)) or price_mkt <= 0:
                    price_mkt = last

            if (not np.isfinite(price_mkt)) or (price_mkt <= 0):
                stats["dropped_bad_price"] += 1
                continue

            rows.append({"K": float(K), "T": float(T), "price_mkt": float(price_mkt), "type": option_side})
            stats["kept"] += 1

    df_opts = pd.DataFrame(rows)
    if df_opts.empty:
        raise ValueError(
            "Aucun point options après filtres.\n"
            "→ Essaye d’élargir la fenêtre de strikes, augmenter le nombre d’expiries, "
            "ou décocher 'mid-only'."
        )

    return df_opts, S_spot, stats


def simulate_options_df(S: float, r: float, option_type: str) -> pd.DataFrame:
    """
    Démo robuste : génère un set (K, T, price_mkt) avec des prix positifs
    (pas forcément arbitrage-free strict, mais stable et utilisable pour construire une surface).
    """
    Ks = np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * S
    Ts = np.array([0.25, 0.5, 1.0, 2.0], dtype=float)

    rows = []
    for T in Ts:
        for K in Ks:
            # vol stylisée (smile + term)
            true_sigma = 0.20 + 0.15 * ((K - S) / max(S, 1e-9)) ** 2 + 0.03 * np.sqrt(T)

            # prix "marché" simulé via BS pour être cohérent
            bs = BlackScholesModel()
            p = OptionParams(S=S, K=float(K), T=float(T), r=float(r), option_type=option_type)
            price_mkt = bs.price(p, float(true_sigma))

            rows.append({"K": float(K), "T": float(T), "price_mkt": float(max(price_mkt, 0.01)), "type": option_type})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Inputs globaux + choix source A/B
# ---------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    S_input = st.number_input("S (Spot) — utilisé pour la simulation", value=100.0, min_value=0.0)
with c2:
    r = st.number_input("r (taux) — input utilisateur", value=0.02, step=0.005)
with c3:
    method = st.selectbox("Méthode calibration", ["brent", "newton"])

st.divider()

source = st.radio("Source des options", ["A) Simulées", "B) Yahoo Finance"], horizontal=True)

st.subheader("Paramètres calibration")
cc1, cc2, cc3, cc4 = st.columns(4)
with cc1:
    sigma0 = st.number_input("σ0 (Newton)", value=0.20, min_value=0.0)
with cc2:
    sigma_min = st.number_input("σ min (Brent)", value=1e-6, format="%.8f")
with cc3:
    sigma_max = st.number_input("σ max (Brent)", value=5.0)
with cc4:
    option_type_default = st.selectbox("Type (pour simu / fallback)", ["call", "put"])

config = VolSurfaceConfig(method=method, sigma0=sigma0, sigma_min=sigma_min, sigma_max=sigma_max)
builder = VolSurfaceBuilder(config=config)

st.divider()


# ---------------------------------------------------------------------
# Construire df_opts (A ou B)
# ---------------------------------------------------------------------
if source.startswith("A"):
    st.subheader("A) Jeu simulé")
    st.write("Démo robuste : génération d’un set d’options simulé puis surface σ_imp(K,T).")

    df_opts = simulate_options_df(S=float(S_input), r=float(r), option_type=option_type_default)
    S_used = float(S_input)

else:
    st.subheader("B) Yahoo Finance (options chain)")
    ticker = st.text_input("Ticker Yahoo (ex: AAPL, MSFT, SPY)", value="AAPL")
    option_side = st.selectbox("Type d’option à extraire", ["call", "put"])
    max_expiries = st.slider("Nombre de maturités à extraire", min_value=1, max_value=10, value=5)

    st.markdown("### Filtres qualité (Yahoo)")
    q1, q2, q3 = st.columns(3)
    with q1:
        use_mid_only = st.checkbox("Utiliser uniquement le mid (bid/ask > 0)", value=True)
    with q2:
        min_days = st.slider("Maturité min (jours)", min_value=0, max_value=60, value=7)
    with q3:
        strike_window = st.slider("Fenêtre strikes autour de S", 0.3, 2.0, (0.7, 1.3), 0.05)

    if st.button("Extraire depuis Yahoo"):
        try:
            df_opts, S_spot, stats = yahoo_extract_options(
                ticker=ticker,
                option_side=option_side,
                max_expiries=max_expiries,
                use_mid_only=use_mid_only,
                strike_min_mult=float(strike_window[0]),
                strike_max_mult=float(strike_window[1]),
                min_days_to_expiry=int(min_days),
            )
            st.session_state["df_opts_yahoo"] = df_opts
            st.session_state["S_used_yahoo"] = float(S_spot)
            st.session_state["stats_yahoo"] = stats
            st.success(f"Extraction OK — Spot ~ {S_spot:.4f} — Points gardés: {stats['kept']}")
        except Exception as e:
            st.error(str(e))
            st.stop()

    if "df_opts_yahoo" not in st.session_state:
        st.warning("Clique sur **Extraire depuis Yahoo** pour charger les options.")
        st.stop()

    df_opts = st.session_state["df_opts_yahoo"]
    S_used = st.session_state.get("S_used_yahoo", float("nan"))

    stats = st.session_state.get("stats_yahoo", {})
    if stats:
        st.write("**Stats filtres Yahoo :**", stats)

# Affichage entrée standardisée
st.subheader("Données options (entrée standardisée)")
st.write(f"Spot utilisé pour la calibration : **S = {S_used:.4f}**")
st.dataframe(df_opts.head(50), use_container_width=True)
st.caption(f"Nombre total de points options : {len(df_opts)}")

st.divider()


# ---------------------------------------------------------------------
# Construire surface σ_imp(K,T)
# ---------------------------------------------------------------------
if st.button("Construire σ_imp(K, T)"):
    if "type" not in df_opts.columns:
        df_opts = df_opts.copy()
        df_opts["type"] = option_type_default

    try:
        df_iv = builder.compute_iv_table(df_opts, S=float(S_used), r=float(r), default_type=option_type_default)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.subheader("Table σ_imp")
    st.dataframe(df_iv, use_container_width=True)

    df_iv_clean = df_iv.dropna(subset=["sigma_imp"])
    if df_iv_clean.empty:
        st.error("Tous les points sont NaN après calibration. Essaye d'élargir les bornes σ ou d'ajuster les filtres Yahoo.")
        st.stop()

    surf = builder.pivot_surface(df_iv_clean)

    st.subheader("Matrice σ_imp (T x K)")
    st.dataframe(surf, use_container_width=True)

    # Heatmap
    st.subheader("Heatmap σ_imp")
    fig_hm = go.Figure(
        data=go.Heatmap(
            x=surf.columns.values,
            y=surf.index.values,
            z=surf.values,
            colorbar=dict(title="σ_imp"),
        )
    )
    fig_hm.update_layout(xaxis_title="Strike K", yaxis_title="Maturité T")
    st.plotly_chart(fig_hm, use_container_width=True)

    # Surface 3D (pivot)
    st.subheader("Surface 3D σ_imp(K,T) (points pivot)")
    X = surf.columns.values
    Y = surf.index.values
    Z = surf.values

    fig_surf = go.Figure(data=[go.Surface(x=X, y=Y, z=Z)])
    fig_surf.update_layout(
        scene=dict(xaxis_title="K", yaxis_title="T", zaxis_title="σ_imp"),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    st.plotly_chart(fig_surf, use_container_width=True)
