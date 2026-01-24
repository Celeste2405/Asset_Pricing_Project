import sys
from pathlib import Path
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Pour que "from model..." marche quel que soit le dossier
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from model.equity_model import (
    bs_call_price, bs_put_price,
    bs_call_delta, bs_call_gamma, bs_call_vega, bs_call_theta,
    bs_put_delta, bs_put_gamma, bs_put_vega, bs_put_theta,
    payoff_call, payoff_put
)
from model.mc_vanilla_model import VanillaOptionParams, VanillaMCPricer
from model.market_data import list_expiries, get_option_chain, get_spot_price
from ui.page_docs import show_page_docs

st.set_page_config(page_title="Pricing", layout="wide")
st.title("1) Pricing du produit (Prix + Greeks + Payoff)")
show_page_docs("pricing")

pricer_mc = VanillaMCPricer()

# ✅ IMPORTANT : toujours initialiser price_mkt
price_mkt = None

# =========================================================
# Inputs + mode Yahoo
# =========================================================
col1, col2 = st.columns(2)

with col1:
    input_mode = st.selectbox("Source des paramètres", ["Manuel", "Option Yahoo (vanilla)"])
    option_type = st.selectbox("Type de produit", ["Call européen", "Put européen"])
    model = st.selectbox("Modèle", ["Black-Scholes", "Monte Carlo (Vanilla)"])

    # Defaults
    S_default, K_default, T_default = 100.0, 100.0, 1.0

    if input_mode == "Option Yahoo (vanilla)":
        st.subheader("Sélection d'une option (Yahoo Finance)")
        ticker = st.text_input("Ticker (ex: AAPL, MSFT, ^FCHI)", value="AAPL")

        try:
            exps = list_expiries(ticker)
        except Exception as e:
            st.error("Impossible de récupérer les maturités Yahoo. Vérifiez yfinance / connexion / ticker.")
            st.exception(e)
            st.stop()

        if not exps:
            st.warning("Aucune maturité trouvée pour ce ticker.")
            st.stop()

        expiry = st.selectbox("Maturité (expiry)", exps, index=0)

        try:
            spot = get_spot_price(ticker)
        except Exception:
            spot = None

        if spot is not None:
            st.caption(f"Spot estimé (dernier close) : S ≈ {spot:.4f}")
            S_default = float(spot)

        require_bid_ask = st.checkbox("Exiger bid/ask valides (plus propre)", value=True)
        min_oi = st.slider("Open interest min", 0, 5000, 0, step=50)
        min_vol = st.slider("Volume min", 0, 5000, 0, step=50)

        opt_type = "call" if option_type == "Call européen" else "put"

        try:
            df_chain = get_option_chain(
                ticker=ticker,
                expiry=expiry,
                option_type=opt_type,
                min_volume=int(min_vol),
                min_open_interest=int(min_oi),
                require_bid_ask=bool(require_bid_ask),
            )
        except Exception as e:
            st.error("Erreur en récupérant la chaîne d'options Yahoo.")
            st.exception(e)
            st.stop()

        if df_chain.empty:
            st.warning("Aucune option ne passe les filtres. Essayez de relâcher les filtres.")
            st.stop()

        strikes = df_chain["K"].astype(float).tolist()
        idx_atm = int(np.argmin(np.abs(np.array(strikes) - S_default)))
        K_choice = st.selectbox("Strike K", strikes, index=idx_atm)

        row = df_chain[df_chain["K"] == float(K_choice)].iloc[0]

        K_default = float(row["K"])
        T_default = float(row["T"])
        price_mkt = float(row["mid"])

        st.success(f"Prix marché (mid) sélectionné : {price_mkt:.6f}")
        st.caption(f"(bid={float(row['bid']):.6f} | ask={float(row['ask']):.6f} | last={float(row['last']):.6f})")

        with st.expander("Voir la chaîne filtrée (extrait)"):
            st.dataframe(df_chain[["type", "K", "mid", "bid", "ask", "last", "volume", "openInterest"]].head(30))

    # Inputs finaux
    S = st.number_input("S (Spot)", value=float(S_default), min_value=0.0)
    K = st.number_input("K (Strike)", value=float(K_default), min_value=0.0)
    T = st.number_input("T (Maturité en années)", value=float(T_default), min_value=0.0)

with col2:
    r = st.number_input("r (Taux sans risque) — input utilisateur", value=0.02, step=0.005)
    sigma = st.number_input("σ (Volatilité)", value=0.20, min_value=0.0, step=0.01)

    payoff_mult = st.slider(
        "Plage pour afficher le payoff autour de S (ex: 0.5S → 1.5S)",
        min_value=1.1, max_value=3.0, value=1.5, step=0.1
    )

st.divider()

# -------------------------
# Paramètres Monte Carlo
# -------------------------
if model.startswith("Monte Carlo"):
    st.subheader("Paramètres Monte Carlo")
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        n_paths = st.select_slider("n_paths", options=[5_000, 10_000, 20_000, 50_000, 80_000, 120_000], value=50_000)
    with mc2:
        n_steps = st.slider("n_steps", min_value=10, max_value=365, value=252)
    with mc3:
        antithetic = st.checkbox("Antithetic variates", value=True)
    with mc4:
        seed = st.number_input("Seed", value=42, step=1)
else:
    n_paths, n_steps, antithetic, seed = 50_000, 252, True, 42

st.divider()

if st.button("Calculer le prix + greeks + payoff"):
    is_call = (option_type == "Call européen")
    opt_type_mc = "call" if is_call else "put"

    # ✅ on sauvegarde toujours les inputs (handoff calibration)
    st.session_state["last_pricing_inputs"] = {
        "option_type": "call" if is_call else "put",
        "S": float(S),
        "K": float(K),
        "T": float(T),
        "r": float(r),
        "sigma": float(sigma),
    }
    if price_mkt is not None:
        st.session_state["price_mkt"] = float(price_mkt)

    # =========================================================
    # 1) PRIX
    # =========================================================
    if model == "Black-Scholes":
        price = bs_call_price(S, K, T, r, sigma) if is_call else bs_put_price(S, K, T, r, sigma)

        # ✅ save result
        st.session_state["last_pricing_result"] = {"model": "Black-Scholes", "price": float(price)}

        st.subheader("Résultat Pricing (Black–Scholes)")
        st.metric("Prix (modèle)", f"{price:.6f}")

        if price_mkt is not None:
            st.metric("Prix marché (mid)", f"{price_mkt:.6f}")
            st.metric("Écart (modèle - marché)", f"{(price - price_mkt):.6f}")

        # Greeks analytiques
        if is_call:
            delta = bs_call_delta(S, K, T, r, sigma)
            gamma = bs_call_gamma(S, K, T, r, sigma)
            vega  = bs_call_vega(S, K, T, r, sigma)
            theta = bs_call_theta(S, K, T, r, sigma)
        else:
            delta = bs_put_delta(S, K, T, r, sigma)
            gamma = bs_put_gamma(S, K, T, r, sigma)
            vega  = bs_put_vega(S, K, T, r, sigma)
            theta = bs_put_theta(S, K, T, r, sigma)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Delta", f"{delta:.6f}")
        c2.metric("Gamma", f"{gamma:.6f}")
        c3.metric("Vega",  f"{vega:.6f}")
        c4.metric("Theta", f"{theta:.6f}")

    else:
        p = VanillaOptionParams(
            S=float(S), K=float(K), T=float(T), r=float(r), sigma=float(sigma),
            option_type=opt_type_mc,
            n_steps=int(n_steps),
            n_paths=int(n_paths),
            antithetic=bool(antithetic),
            seed=int(seed),
        )
        res = pricer_mc.price(p)

        # ✅ save result
        st.session_state["last_pricing_result"] = {"model": "Monte Carlo", "price": float(res.price)}

        st.subheader("Résultat Pricing (Monte Carlo Vanilla)")
        cA, cB, cC = st.columns(3)
        cA.metric("Prix MC", f"{res.price:.6f}")
        cB.metric("Std. error", f"{res.stderr:.6f}")
        cC.metric("IC 95%", f"[{res.ci_low:.6f} ; {res.ci_high:.6f}]")

        if price_mkt is not None:
            st.metric("Prix marché (mid)", f"{price_mkt:.6f}")
            st.metric("Écart (MC - marché)", f"{(res.price - price_mkt):.6f}")

        # Comparaison BS
        bs_ref = bs_call_price(S, K, T, r, sigma) if is_call else bs_put_price(S, K, T, r, sigma)
        st.divider()
        st.subheader("Comparaison Monte Carlo vs Black–Scholes")
        d1, d2, d3 = st.columns(3)
        d1.metric("Prix BS (référence)", f"{bs_ref:.6f}")
        d2.metric("Prix MC", f"{res.price:.6f}")
        d3.metric("Écart (MC - BS)", f"{(res.price - bs_ref):.6f}")

        # Greeks MC
        st.divider()
        st.subheader("Greeks Monte Carlo (différences finies)")
        eps1, eps2, eps3 = st.columns(3)
        with eps1:
            eps_S = st.number_input("ε_S (Delta)", value=0.2, min_value=1e-6)
        with eps2:
            eps_sigma = st.number_input("ε_σ (Vega)", value=0.002, min_value=1e-6)
        with eps3:
            eps_r = st.number_input("ε_r (Rho)", value=0.0001, min_value=1e-8)

        g = pricer_mc.greeks_finite_diff(
            p,
            eps_S=float(eps_S),
            eps_sigma=float(eps_sigma),
            eps_r=float(eps_r),
            eps_T=1e-3,
        )

        gc1, gc2, gc3, gc4 = st.columns(4)
        gc1.metric("Delta", f"{g['delta']:.6f}")
        gc2.metric("Vega",  f"{g['vega']:.6f}")
        gc3.metric("Rho",   f"{g['rho']:.6f}")
        gc4.metric("Theta (∂V/∂T)", f"{g['theta']:.6f}")

    # =========================================================
    # 3) PAYOFF
    # =========================================================
    st.divider()
    st.subheader("Payoff à maturité")

    ST = np.linspace(max(0.0, S / payoff_mult), S * payoff_mult, 400)
    payoff = payoff_call(ST, K) if is_call else payoff_put(ST, K)
    payoff_label = "Payoff Call" if is_call else "Payoff Put"

    fig = plt.figure()
    plt.plot(ST, payoff, label=payoff_label)
    plt.axvline(K, linestyle="--", label="Strike K")
    plt.title("Payoff à maturité")
    plt.xlabel("S_T")
    plt.ylabel("Payoff")
    plt.legend()
    st.pyplot(fig)
