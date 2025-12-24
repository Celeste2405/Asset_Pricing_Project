import sys
from pathlib import Path
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.calibration_model import OptionParams, BlackScholesModel, ImpliedVolCalibrator

st.set_page_config(page_title="Calibration", layout="wide")
st.title("2) Calibration : Volatilité implicite (ω_imp / σ_imp)")

st.write(
    "Tu donnes les paramètres + un **prix marché** et tu choisis la méthode de calibration. "
    "On renvoie **σ_imp** (omega imp)."
)

model = BlackScholesModel()
calibrator = ImpliedVolCalibrator(model)

col1, col2 = st.columns(2)

with col1:
    option_type = st.selectbox("Type", ["call", "put"])
    method = st.selectbox("Méthode", ["brent", "newton"])

    S = st.number_input("S (Spot)", value=100.0, min_value=0.0)
    K = st.number_input("K (Strike)", value=100.0, min_value=0.0)
    T = st.number_input("T (années)", value=1.0, min_value=0.0)

with col2:
    r = st.number_input("r (taux) — input utilisateur", value=0.02, step=0.005)
    market_price = st.number_input("Prix marché de l'option", value=10.0, min_value=0.0)

    sigma0 = st.number_input("σ initial (Newton)", value=0.20, min_value=0.0)
    sigma_min = st.number_input("σ min (Brent)", value=1e-6, format="%.8f")
    sigma_max = st.number_input("σ max (Brent)", value=5.0)

st.divider()

if st.button("Calibrer σ_imp"):
    p = OptionParams(S=S, K=K, T=T, r=r, option_type=option_type)

    sigma_imp = calibrator.implied_vol(
        market_price=market_price,
        p=p,
        method=method,
        sigma0=sigma0,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
    )

    st.subheader("Résultat")

    if np.isnan(sigma_imp):
        st.error("Calibration impossible (prix incompatible / bornes σ insuffisantes / vega trop faible).")
        st.stop()

    st.metric("ω_imp (σ_imp)", f"{sigma_imp:.6f}")

    # Check : prix BS avec sigma_imp
    price_check = model.price(p, sigma_imp)
    st.write(f"Prix BS avec σ_imp : **{price_check:.6f}** (doit ≈ prix marché)")
    st.write(f"Écart (BS - marché) : **{(price_check - market_price):.6e}**")
