import sys
from pathlib import Path
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.calibration_model import OptionParams, BlackScholesModel, ImpliedVolCalibrator
from ui.page_docs import show_page_docs

st.set_page_config(page_title="CALIBRATION", layout="wide")
st.title("Calibration : Volatilité implicite (ω_imp / σ_imp)")
show_page_docs("calibration")

st.write(
    "Entrez les paramètres et un **prix marché** et choisissez la méthode de calibration. "
    "L'application renvoie **σ_imp** la volatilité implicite."
)

# =========================================================
# Init session state (valeurs par défaut des champs)
# =========================================================
if "calib_S" not in st.session_state:
    st.session_state["calib_S"] = 100.0
if "calib_K" not in st.session_state:
    st.session_state["calib_K"] = 100.0
if "calib_T" not in st.session_state:
    st.session_state["calib_T"] = 1.0
if "calib_r" not in st.session_state:
    st.session_state["calib_r"] = 0.02
if "calib_option_type" not in st.session_state:
    st.session_state["calib_option_type"] = "call"
if "calib_market_price" not in st.session_state:
    st.session_state["calib_market_price"] = 10.0

# =========================================================
# Pré-remplissage depuis Pricing
# (suppose que la page Pricing stocke last_pricing_inputs / last_pricing_result)
# =========================================================
col_prefill1, col_prefill2 = st.columns([1, 2])

with col_prefill1:
    if st.button("Pré-remplir depuis Pricing"):
        payload = st.session_state.get("last_pricing_inputs", None)
        payload_price = st.session_state.get("last_pricing_result", None)

        if payload is None:
            st.warning("Aucune info trouvée depuis la page Pricing (lance un pricing avant).")
        else:
            st.session_state["calib_S"] = float(payload.get("S", st.session_state["calib_S"]))
            st.session_state["calib_K"] = float(payload.get("K", st.session_state["calib_K"]))
            st.session_state["calib_T"] = float(payload.get("T", st.session_state["calib_T"]))
            st.session_state["calib_r"] = float(payload.get("r", st.session_state["calib_r"]))
            st.session_state["calib_option_type"] = str(payload.get("option_type", st.session_state["calib_option_type"]))

            # Option : préremplir le prix marché avec le prix obtenu dans Pricing
            if payload_price is not None and "price" in payload_price:
                st.session_state["calib_market_price"] = float(payload_price["price"])

            st.success("Champs pré-remplis")

with col_prefill2:
    st.caption("Lancez un pricing puis cliquez ici pour réutiliser les mêmes paramètres.")

st.divider()

# =========================================================
# Inputs calibration
# =========================================================
model = BlackScholesModel()
calibrator = ImpliedVolCalibrator(model)

col1, col2 = st.columns(2)

with col1:
    option_type = st.selectbox("Type", ["call", "put"], key="calib_option_type")
    method = st.selectbox("Méthode", ["brent", "newton"])

    S = st.number_input("S (Spot)", min_value=0.0, key="calib_S")
    K = st.number_input("K (Strike)", min_value=0.0, key="calib_K")
    T = st.number_input("T (années)", min_value=0.0, key="calib_T")

with col2:
    r = st.number_input("r (Taux)", step=0.005, key="calib_r")
    market_price = st.number_input("Prix marché de l'option", min_value=0.0, key="calib_market_price")

    sigma0 = st.number_input("σ initial (Newton)", value=0.20, min_value=0.0)
    sigma_min = st.number_input("σ min (Brent)", value=1e-6, format="%.8f")
    sigma_max = st.number_input("σ max (Brent)", value=5.0)

# Optionnel : bruit pédagogique (évite BS-marché = 0 si tu utilises le prix BS comme "marché")
st.caption("Optionnel : pour une démonstration, vous pouvez perturber légèrement le prix marché.")
add_noise = st.checkbox("Ajouter un bruit au prix marché", value=False)
noise_pct = st.slider("Bruit (%)", 0.0, 5.0, 0.0, step=0.5) / 100.0
market_price_used = float(market_price) * (1.0 + noise_pct) if add_noise else float(market_price)
if add_noise and noise_pct > 0:
    st.info(f"Prix marché utilisé (perturbé) : {market_price_used:.6f}")

st.divider()

# =========================================================
# Calibration
# =========================================================
if st.button("Calibrer σ_imp"):
    p = OptionParams(S=float(S), K=float(K), T=float(T), r=float(r), option_type=str(option_type))

    sigma_imp = calibrator.implied_vol(
        market_price=float(market_price_used),
        p=p,
        method=str(method),
        sigma0=float(sigma0),
        sigma_min=float(sigma_min),
        sigma_max=float(sigma_max),
    )

    st.subheader("Résultat")

    if sigma_imp is None or np.isnan(sigma_imp):
        st.error("Calibration impossible (prix incompatible / bornes σ insuffisantes / vega trop faible).")
        st.stop()

    st.metric("ω_imp (σ_imp)", f"{float(sigma_imp):.6f}")

    # Check : prix BS avec sigma_imp
    price_check = model.price(p, float(sigma_imp))
    st.write(f"Prix BS avec σ_imp : **{price_check:.6f}** (doit ≈ prix marché)")
    st.write(f"Écart (BS - marché) : **{(price_check - float(market_price_used)):.6e}**")

    st.divider()

    # Sauvegarde vers session_state
    st.session_state["sigma_imp"] = float(sigma_imp)
    st.session_state["sigma_imp_source"] = {
        "S": float(S),
        "K": float(K),
        "T": float(T),
        "r": float(r),
        "option_type": str(option_type),
        "market_price": float(market_price_used),
        "method": str(method),
    }

    st.success("σ_imp sauvegardée dans la session (vous pouvez la réutiliser sur d'autres pages).")
