import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ui.page_docs import show_page_docs
from model.vol_surface_data import YahooSurfaceConfig, build_yahoo_option_dataset
from model.vol_surface_builder import SurfaceBuildConfig, VolSurfaceBuilder
from model.equity_model import bs_call_price, bs_put_price
from model.sabr_model import SABRCalibrator, sabr_iv_vector
from model.dupire_model import DupireLocalVol, DupireConfig

# =========================================================
# Page config
# =========================================================
st.set_page_config(page_title="Vol Surface", layout="wide")
st.title("3) Vol Surface : σ_imp(K, T)")
show_page_docs("vol_surface")

builder = VolSurfaceBuilder()

# =========================================================
# Session state init (OBLIGATOIRE)
# =========================================================
def init_state():
    defaults = {
        "vs_df_opts": None,        # dataset options (Yahoo)
        "vs_S": None,              # spot (Yahoo)
        "vol_surface_df": None,    # surface sigma_imp points
        "vol_surface_mat": None,   # matrice (T x K)
        "vol_surface_meta": None,  # dict meta
        "vs_last_signature": None  # pour reset auto
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# =========================================================
# Helpers
# =========================================================
def reset_surface():
    st.session_state["vol_surface_df"] = None
    st.session_state["vol_surface_mat"] = None
    st.session_state["vol_surface_meta"] = None

def signature(*items):
    # signature légère pour détecter changements majeurs
    return str(items)


# =========================================================
# A) Choix source + paramètres globaux
# =========================================================
source = st.radio(
    "Source de données",
    ["A) Données simulées", "B) Yahoo Finance (réel)"],
    horizontal=True
)

colL, colR = st.columns(2)
with colL:
    r = st.number_input("r (Taux)", value=0.02, step=0.005)
    option_type = st.selectbox("Type d'option", ["call", "put"])
    method = st.selectbox("Méthode σ_imp", ["brent", "newton"])

with colR:
    sigma0 = st.number_input("σ0 (Newton)", value=0.20, min_value=0.0)
    sigma_min = st.number_input("σ_min (Brent)", value=1e-6, format="%.8f")
    sigma_max = st.number_input("σ_max (Brent)", value=5.0)

st.divider()

# =========================================================
# B) Construction df options (A ou B)
# =========================================================
df_opts = None
S = None

if source.startswith("A)"):
    st.subheader("A) Génération d'un dataset simulé (marché synthétique)")
    S = st.number_input("Spot S (simulé)", value=100.0, min_value=1e-6)

    Ks = np.array(
        st.multiselect(
            "Strikes K",
            [80.0, 90.0, 100.0, 110.0, 120.0],
            default=[80.0, 90.0, 100.0, 110.0, 120.0]
        ),
        dtype=float
    )
    Ts = np.array(
        st.multiselect(
            "Maturités T (années)",
            [0.25, 0.5, 1.0, 2.0],
            default=[0.25, 0.5, 1.0, 2.0]
        ),
        dtype=float
    )

    if Ks.size == 0 or Ts.size == 0:
        st.warning("Choisissez au moins un strike et une maturité.")
        st.stop()

    # si tu changes ces éléments => reset surface
    sig = signature("SIM", float(S), tuple(Ks.tolist()), tuple(Ts.tolist()), option_type, float(r), method)
    if st.session_state["vs_last_signature"] != sig:
        st.session_state["vs_last_signature"] = sig
        reset_surface()

    rows = []
    for T_ in Ts:
        for K_ in Ks:
            # vol vraie stylisée (smile + term structure)
            true_sigma = 0.20 + 0.15 * ((K_ - S) / max(S, 1e-9)) ** 2 + 0.03 * np.sqrt(T_)
            if option_type == "call":
                price_mkt = bs_call_price(S, K_, T_, r, true_sigma)
            else:
                price_mkt = bs_put_price(S, K_, T_, r, true_sigma)
            rows.append({"K": float(K_), "T": float(T_), "price_mkt": float(price_mkt)})

    df_opts = pd.DataFrame(rows)

else:
    st.subheader("B) Dataset Yahoo Finance (options réelles)")
    ticker = st.text_input("Ticker (ex: AAPL, MSFT, ^FCHI)", value="AAPL")

    max_exp = st.slider("Nombre max de maturités (expiries) à charger", 1, 12, 6)
    m_low = st.slider("Moneyness low (K >= low*S)", 0.3, 0.95, 0.7)
    m_high = st.slider("Moneyness high (K <= high*S)", 1.05, 2.0, 1.3)

    require_bid_ask = st.checkbox("Exiger bid/ask valides", value=True)
    min_oi = st.slider("Open interest min", 0, 5000, 0, step=50)
    min_vol = st.slider("Volume min", 0, 5000, 0, step=50)

    # si tu changes ces éléments => reset surface (car dataset change)
    sig = signature("YAHOO", ticker, int(max_exp), float(m_low), float(m_high), bool(require_bid_ask),
                    int(min_oi), int(min_vol), option_type, float(r), method)
    if st.session_state["vs_last_signature"] != sig:
        st.session_state["vs_last_signature"] = sig
        reset_surface()

    cfg_y = YahooSurfaceConfig(
        ticker=ticker,
        option_type=option_type,
        max_expiries=int(max_exp),
        moneyness_low=float(m_low),
        moneyness_high=float(m_high),
        require_bid_ask=bool(require_bid_ask),
        min_open_interest=int(min_oi),
        min_volume=int(min_vol),
    )

    c1, c2 = st.columns([1, 2])
    with c1:
        load_btn = st.button("Charger les options Yahoo")
    with c2:
        if st.session_state["vs_df_opts"] is not None and st.session_state["vs_S"] is not None:
            st.success(f"Dataset en mémoire: {len(st.session_state['vs_df_opts'])} options (S≈{st.session_state['vs_S']:.4f})")

    if load_btn:
        try:
            df_loaded, S_loaded = build_yahoo_option_dataset(cfg_y)
            st.session_state["vs_df_opts"] = df_loaded
            st.session_state["vs_S"] = float(S_loaded)
            reset_surface()
            st.success(f"Dataset chargé ✅ {len(df_loaded)} options. Spot S≈{S_loaded:.4f}")
        except Exception as e:
            st.error("Impossible de construire le dataset Yahoo (filtres trop stricts / ticker / connexion).")
            st.exception(e)

    # on lit depuis session_state si présent
    df_opts = st.session_state["vs_df_opts"]
    S = st.session_state["vs_S"]

st.divider()

if df_opts is None or S is None:
    st.info("Construisez un dataset (simulé ou Yahoo) pour continuer.")
    st.stop()

# =========================================================
# C) Affichage dataset
# =========================================================
st.subheader("Dataset d'options")
st.write(f"Spot utilisé : **S = {float(S):.6f}**")
st.dataframe(df_opts.head(50))

st.divider()

# =========================================================
# D) Calcul σ_imp
# =========================================================
cfg = SurfaceBuildConfig(
    r=float(r),
    option_type=option_type,
    method=method,
    sigma0=float(sigma0),
    sigma_min=float(sigma_min),
    sigma_max=float(sigma_max),
)

build_btn = st.button("Construire la surface σ_imp (calibration point par point)")
if build_btn:
    with st.spinner("Calibration σ_imp en cours..."):
        df_surface = builder.build_surface(df_opts=df_opts, S=float(S), cfg=cfg)
        mat = builder.pivot_surface(df_surface)

    st.session_state["vol_surface_df"] = df_surface
    st.session_state["vol_surface_mat"] = mat
    st.session_state["vol_surface_meta"] = {"S": float(S), "r": float(r), "option_type": option_type}
    st.success(f"Surface construite ✅ Points gardés: {len(df_surface)}")

# lecture surface depuis session
df_surface = st.session_state["vol_surface_df"]
mat = st.session_state["vol_surface_mat"]

if df_surface is None or mat is None:
    st.info("Cliquez sur **Construire la surface σ_imp** pour calculer les volatilités implicites.")
    st.stop()

# =========================================================
# E) Affichage surface
# =========================================================
st.subheader("Table σ_imp (points)")

df_surface = st.session_state["vol_surface_df"]
mat = st.session_state["vol_surface_mat"]

if df_surface is None or mat is None:
    st.info("👉 Construisez d'abord la surface σ_imp en cliquant sur le bouton ci-dessus.")
    st.stop()

st.subheader("Table σ_imp")

st.dataframe(df_surface.head(50))

st.divider()
st.subheader("SABR : lissage du smile (par maturité)")

beta = st.slider("β (fixé)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
max_iter = st.slider("Itérations max calibration", 50, 500, 200, step=50)

if st.button("Calibrer SABR sur chaque maturité"):
    cal = SABRCalibrator(beta=float(beta))

    Ts = np.array(mat.index, dtype=float)
    Ks = np.array(mat.columns, dtype=float)

    sabr_mat = np.full_like(mat.values, np.nan, dtype=float)
    params_rows = []

    # F approx = S (simplification)
    S_used = float(st.session_state["vol_surface_meta"]["S"]) if st.session_state["vol_surface_meta"] else float(df_surface["S"].iloc[0])
    F = S_used

    with st.spinner("Calibration SABR en cours..."):
        for i, T_ in enumerate(Ts):
            iv_mkt = mat.values[i, :].astype(float)

            # garder strikes où iv dispo
            mask = np.isfinite(iv_mkt)
            Ks_i = Ks[mask]
            iv_i = iv_mkt[mask]

            if Ks_i.size < 3:
                continue

            est, loss = cal.calibrate(F=F, Ks=Ks_i, T=float(T_), iv_mkt=iv_i, max_iter=int(max_iter))
            params_rows.append({"T": float(T_), "alpha": est.alpha, "beta": est.beta, "rho": est.rho, "nu": est.nu, "loss": loss})

            # reconstruire smile sur tous Ks
            sabr_smile = sabr_iv_vector(F=F, Ks=Ks, T=float(T_), p=est)
            sabr_mat[i, :] = sabr_smile

    sabr_df_params = pd.DataFrame(params_rows).sort_values("T").reset_index(drop=True)
    sabr_mat_df = pd.DataFrame(sabr_mat, index=mat.index, columns=mat.columns)

    st.session_state["sabr_params_df"] = sabr_df_params
    st.session_state["sabr_mat"] = sabr_mat_df

    st.success("SABR calibré ✅")

# affichage si dispo
if "sabr_mat" in st.session_state and st.session_state["sabr_mat"] is not None:
    st.subheader("Paramètres SABR par maturité")
    st.dataframe(st.session_state["sabr_params_df"])

    st.subheader("Matrice σ_SABR (T x K)")
    st.dataframe(st.session_state["sabr_mat"])

    st.divider()
    st.subheader("Heatmap σ_SABR")

    sabr_mat_df = st.session_state["sabr_mat"]
    fig_s = plt.figure()
    plt.imshow(sabr_mat_df.values, aspect="auto", origin="lower")
    plt.xticks(ticks=np.arange(sabr_mat_df.shape[1]), labels=[f"{k:.0f}" for k in sabr_mat_df.columns], rotation=45)
    plt.yticks(ticks=np.arange(sabr_mat_df.shape[0]), labels=[f"{t:.3f}" for t in sabr_mat_df.index])
    plt.colorbar(label="σ_SABR")
    plt.xlabel("Strike K")
    plt.ylabel("Maturité T (années)")
    plt.title("Heatmap σ_SABR(K,T)")
    st.pyplot(fig_s)




# =========================================================
# F) Heatmap
# =========================================================
st.divider()
st.subheader("Heatmap σ_imp")

fig = plt.figure()
arr = mat.values
plt.imshow(arr, aspect="auto", origin="lower")
plt.xticks(ticks=np.arange(mat.shape[1]), labels=[f"{k:.0f}" for k in mat.columns], rotation=45)
plt.yticks(ticks=np.arange(mat.shape[0]), labels=[f"{t:.3f}" for t in mat.index])
plt.colorbar(label="σ_imp")
plt.xlabel("Strike K")
plt.ylabel("Maturité T (années)")
plt.title("Heatmap de la volatilité implicite σ_imp(K,T)")
st.pyplot(fig)

# =========================================================
# G) Surface 3D (optionnel)
# =========================================================
st.divider()
st.subheader("Surface 3D σ_imp (optionnel)")

show_3d = st.checkbox("Afficher surface 3D", value=False)
if show_3d:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Ks = np.array(mat.columns, dtype=float)
    Ts = np.array(mat.index, dtype=float)
    KK, TT = np.meshgrid(Ks, Ts)
    ZZ = mat.values

    fig3 = plt.figure()
    ax = fig3.add_subplot(111, projection="3d")
    ax.plot_surface(KK, TT, ZZ)
    ax.set_xlabel("K")
    ax.set_ylabel("T")
    ax.set_zlabel("σ_imp")
    ax.set_title("Surface σ_imp(K,T)")
    st.pyplot(fig3)


st.divider()
st.subheader("Dupire : volatilité locale σ_loc(K,T)")

use_sabr = False
if "sabr_mat" in st.session_state and st.session_state["sabr_mat"] is not None:
    use_sabr = st.checkbox("Utiliser la surface SABR (plus lisse) si disponible", value=True)

mat_iv_used = st.session_state["sabr_mat"] if (use_sabr and "sabr_mat" in st.session_state and st.session_state["sabr_mat"] is not None) else mat

meta = st.session_state.get("vol_surface_meta", None)
S_used = float(meta["S"]) if meta and "S" in meta else float(df_surface["S"].iloc[0])
r_used = float(meta["r"]) if meta and "r" in meta else float(r)

c1, c2, c3 = st.columns(3)
with c1:
    denom_floor = st.number_input("denom_floor", value=1e-10, format="%.1e")
with c2:
    vol_floor = st.number_input("vol_floor", value=1e-6, format="%.1e")
with c3:
    vol_cap = st.number_input("vol_cap", value=5.0)

if st.button("Calculer σ_loc via Dupire"):
    dup = DupireLocalVol()
    cfg_d = DupireConfig(r=float(r_used), denom_floor=float(denom_floor), vol_floor=float(vol_floor), vol_cap=float(vol_cap))

    with st.spinner("Calcul Dupire en cours..."):
        mat_C, mat_loc = dup.local_vol_from_iv_surface(
            S=float(S_used),
            r=float(r_used),
            mat_iv=mat_iv_used,
            cfg=cfg_d
        )

    st.session_state["dupire_call_mat"] = mat_C
    st.session_state["dupire_locvol_mat"] = mat_loc
    st.success("σ_loc calculée ✅")

if "dupire_locvol_mat" in st.session_state and st.session_state["dupire_locvol_mat"] is not None:
    mat_loc = st.session_state["dupire_locvol_mat"]

    st.subheader("Matrice σ_loc (T x K)")
    st.dataframe(mat_loc)

    st.divider()
    st.subheader("Heatmap σ_loc")

    fig_lv = plt.figure()
    arr = mat_loc.values
    plt.imshow(arr, aspect="auto", origin="lower")
    plt.xticks(ticks=np.arange(mat_loc.shape[1]), labels=[f"{k:.0f}" for k in mat_loc.columns], rotation=45)
    plt.yticks(ticks=np.arange(mat_loc.shape[0]), labels=[f"{t:.3f}" for t in mat_loc.index])
    plt.colorbar(label="σ_loc")
    plt.xlabel("Strike K")
    plt.ylabel("Maturité T (années)")
    plt.title("Heatmap de la volatilité locale σ_loc(K,T) (Dupire)")
    st.pyplot(fig_lv)
else:
    st.info("Clique sur **Calculer σ_loc via Dupire** pour afficher la volatilité locale.")
