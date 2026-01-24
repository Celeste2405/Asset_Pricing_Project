import sys
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ui.page_docs import show_page_docs
from model.barrier_mc_model import BarrierParams, BarrierMCPricer


st.set_page_config(page_title="Barrier Options", layout="wide")
st.title("5) Options Barrière (Knock-Out) — Monte Carlo")
show_page_docs("barrier")

pricer = BarrierMCPricer()

col1, col2 = st.columns(2)

with col1:
    option_type = st.selectbox("Option type", ["call", "put"])
    barrier_type = st.selectbox("Barrier type", ["up-and-out", "down-and-out"])

    S = st.number_input("S (Spot)", value=100.0, min_value=1e-6)
    K = st.number_input("K (Strike)", value=100.0, min_value=1e-6)
    H = st.number_input("H (Barrière)", value=120.0, min_value=1e-6)

with col2:
    T = st.number_input("T (années)", value=1.0, min_value=0.0)
    r = st.number_input("r (taux)", value=0.02, step=0.005)
    sigma = st.number_input("σ (vol)", value=0.20, min_value=0.0, step=0.01)
    rebate = st.number_input("Rebate (payé si KO, à maturité)", value=0.0, min_value=0.0)

st.divider()

st.subheader("Paramètres Monte Carlo")
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    n_paths = st.select_slider("n_paths", options=[10_000, 20_000, 50_000, 80_000, 120_000], value=80_000)
with mc2:
    n_steps = st.slider("n_steps (monitoring discret)", 10, 365, 252)
with mc3:
    antithetic = st.checkbox("Antithetic variates", value=True)
with mc4:
    seed = st.number_input("Seed", value=42, step=1)

st.divider()

if st.button("Pricer l'option barrière"):
    # sanity checks
    if barrier_type == "up-and-out" and H <= max(S, 1e-9):
        st.warning("Up-and-out: si H <= S, la barrière peut être touchée très vite (prix peut se rapprocher du rebate).")
    if barrier_type == "down-and-out" and H >= max(S, 1e-9):
        st.warning("Down-and-out: si H >= S, la barrière peut être touchée très vite.")

    p = BarrierParams(
        S=float(S), K=float(K), T=float(T), r=float(r), sigma=float(sigma),
        option_type=option_type,
        barrier_type=barrier_type,
        H=float(H),
        rebate=float(rebate),
        n_steps=int(n_steps),
        n_paths=int(n_paths),
        antithetic=bool(antithetic),
        seed=int(seed),
    )

    res = pricer.price(p)

    st.subheader("Résultat Monte Carlo")
    cA, cB, cC = st.columns(3)
    cA.metric("Prix MC", f"{res.price:.6f}")
    cB.metric("Std. error", f"{res.stderr:.6f}")
    cC.metric("IC 95%", f"[{res.ci_low:.6f} ; {res.ci_high:.6f}]")

    st.caption(
        f"KO rate ≈ {100*res.details['knockout_rate']:.2f}% | "
        f"n_paths_eff={res.details['n_paths_eff']} | n_steps={res.details['n_steps']} | antithetic={res.details['antithetic']}"
    )

    st.divider()
    st.subheader("Schéma payoff (qualitatif)")

    # Payoff vs ST (en réalité dépend du path, donc ce plot est une intuition)
    ST_grid = np.linspace(0.2 * S, 1.8 * S, 400)
    if option_type == "call":
        vanilla = np.maximum(ST_grid - K, 0.0)
    else:
        vanilla = np.maximum(K - ST_grid, 0.0)

    # on montre payoff "si pas KO" vs "si KO"
    fig = plt.figure()
    plt.plot(ST_grid, vanilla, label="Payoff vanilla (si pas KO)")
    plt.plot(ST_grid, np.full_like(ST_grid, rebate), linestyle="--", label="Payoff si KO (rebate)")
    plt.axvline(K, linestyle=":", label="Strike K")
    plt.axvline(H, linestyle="--", label="Barrière H")
    plt.xlabel("S_T")
    plt.ylabel("Payoff")
    plt.title("Payoff barrière : dépend du path (ce graphique est indicatif)")
    plt.legend()
    st.pyplot(fig)
