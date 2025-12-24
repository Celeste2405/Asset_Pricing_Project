import sys
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Pour que "from models..." marche quel que soit le dossier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.equity_model import (
    bs_call_price, bs_put_price,
    bs_call_delta, bs_call_gamma, bs_call_vega, bs_call_theta,
    payoff_call, payoff_put
)

st.set_page_config(page_title="Pricing", layout="wide")
st.title("1) Pricing du produit (Prix + Greeks + Payoff)")

col1, col2 = st.columns(2)

with col1:
    option_type = st.selectbox("Type de produit", ["Call européen", "Put européen"])
    model = st.selectbox("Modèle", ["Black-Scholes"])  # extensible plus tard

    S = st.number_input("S (Spot)", value=100.0, min_value=0.0)
    K = st.number_input("K (Strike)", value=100.0, min_value=0.0)
    T = st.number_input("T (Maturité en années)", value=1.0, min_value=0.0)

with col2:
    r = st.number_input("r (Taux sans risque) — input utilisateur", value=0.02, step=0.005)
    sigma = st.number_input("σ (Volatilité)", value=0.20, min_value=0.0, step=0.01)

    payoff_mult = st.slider(
        "Plage pour afficher le payoff autour de S (ex: 0.5S → 1.5S)",
        min_value=1.1, max_value=3.0, value=1.5, step=0.1
    )

st.divider()

if st.button("Calculer le prix + greeks + payoff"):
    if model != "Black-Scholes":
        st.error("Seul Black-Scholes est implémenté pour le moment.")
        st.stop()

    is_call = (option_type == "Call européen")

    # Prix
    if is_call:
        price = bs_call_price(S, K, T, r, sigma)
    else:
        price = bs_put_price(S, K, T, r, sigma)

    st.subheader("Résultat Pricing")
    st.metric("Prix", f"{price:.6f}")

    # Greeks : tu as déjà les greeks du CALL dans ton fichier
    # Pour rester cohérent, on affiche les greeks du call (et on prévient si put)
    if is_call:
        delta = bs_call_delta(S, K, T, r, sigma)
        gamma = bs_call_gamma(S, K, T, r, sigma)
        vega  = bs_call_vega(S, K, T, r, sigma)
        theta = bs_call_theta(S, K, T, r, sigma)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delta", f"{delta:.6f}")
        c2.metric("Gamma", f"{gamma:.6f}")
        c3.metric("Vega",  f"{vega:.6f}")
        c4.metric("Theta", f"{theta:.6f}")
    else:
        st.warning("Greeks du PUT non implémentés dans ton fichier actuel (on peut les ajouter).")

    st.divider()

    # Payoff plot
    st.subheader("Payoff à maturité")

    ST = np.linspace(max(0.0, S / payoff_mult), S * payoff_mult, 400)
    if is_call:
        payoff = payoff_call(ST, K)
        payoff_label = "Payoff Call"
    else:
        payoff = payoff_put(ST, K)
        payoff_label = "Payoff Put"

    fig = plt.figure()
    plt.plot(ST, payoff, label=payoff_label)
    plt.axvline(K, linestyle="--", label="Strike K")
    plt.title("Payoff à maturité")
    plt.xlabel("S_T")
    plt.ylabel("Payoff")
    plt.legend()
    st.pyplot(fig)
