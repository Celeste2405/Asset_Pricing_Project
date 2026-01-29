import sys
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Options asiatiques", layout="wide")
st.title("Options Asiatiques — Moyenne Arithmétique & Géométrique")

# ------------------------------------------------------------
# Setup imports safely (évite page blanche)
# ------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from ui.page_docs import show_page_docs



try:
    import numpy as np
    import matplotlib.pyplot as plt

    # Arithmétique (ton module existant)
    from model.asian import AsianOptionParams, AsianArithmeticAveragePricer

    # Géométrique (notre module)
    from model.asian_geom_model import asian_geom_closed_form, AsianGeomParams, AsianGeomMCPricer

except Exception as e:
    st.error("Erreur au chargement de la page (import / dépendance).")
    st.exception(e)
    st.stop()

# Doc (nouvelle clé)
show_page_docs("asian_options")

# ------------------------------------------------------------
# Choix du type d'asiatique
# ------------------------------------------------------------
avg_type = st.radio(
    "Type d'option asiatique",
    ["Moyenne arithmétique (MC)", "Moyenne géométrique (Closed-form + MC)"],
    horizontal=True
)

st.divider()

# ------------------------------------------------------------
# Inputs communs
# ------------------------------------------------------------
c1, c2, c3 = st.columns(3)
with c1:
    option_type = st.selectbox("Type", ["call", "put"])
    S = st.number_input("S (Spot)", value=100.0, min_value=1e-6)
    K = st.number_input("K (Strike)", value=100.0, min_value=1e-6)

with c2:
    T = st.number_input("T (années)", value=1.0, min_value=0.0)
    r = st.number_input("r (taux)", value=0.02, step=0.005)
    sigma = st.number_input("σ (vol)", value=0.20, min_value=0.0, step=0.01)

with c3:
    n_steps = st.slider("n_steps (nombre de fixings)", 5, 365, 60)

# ------------------------------------------------------------
# Branch 1 — Asiatique arithmétique (MC uniquement)
# ------------------------------------------------------------
if avg_type.startswith("Moyenne arithmétique"):
    st.subheader("Asiatique arithmétique — Monte Carlo")
    st.write(
        "Payoff dépend de la **moyenne arithmétique** des prix observés.\n\n"
        "- Call: max(avg(S) - K, 0)\n"
        "- Put : max(K - avg(S), 0)\n"
    )

    pricer = AsianArithmeticAveragePricer()

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        n_paths = st.select_slider(
            "n_paths (trajectoires)",
            options=[5_000, 10_000, 20_000, 50_000, 80_000, 120_000],
            value=50_000
        )
    with mc2:
        antithetic = st.checkbox("Antithetic variates", value=True)
    with mc3:
        seed = st.number_input("Seed (reproductibilité)", value=42, step=1)

    include_S0 = st.checkbox("Inclure S0 dans la moyenne", value=True)

    st.divider()

    if st.button("Pricer l'asiatique arithmétique"):
        p = AsianOptionParams(
            S=float(S), K=float(K), T=float(T), r=float(r), sigma=float(sigma),
            option_type=option_type,
            n_steps=int(n_steps),
            n_paths=int(n_paths),
            antithetic=bool(antithetic),
            seed=int(seed),
            include_S0_in_average=bool(include_S0),
        )

        res = pricer.price(p)

        st.subheader("Résultat (Monte Carlo)")
        a, b, c = st.columns(3)
        a.metric("Prix", f"{res.price:.6f}")
        b.metric("Std. error", f"{res.stderr:.6f}")
        c.metric("IC 95%", f"[{res.ci_low:.6f} ; {res.ci_high:.6f}]")

        st.caption(
            f"Détails: n_paths_eff={int(res.details.get('n_paths_eff', 0))}, "
            f"n_steps={int(res.details.get('n_steps', 0))}, "
            f"antithetic={int(res.details.get('antithetic', 0))}"
        )

        st.divider()
        st.subheader("Greeks (différences finies)")

        eps1, eps2, eps3 = st.columns(3)
        with eps1:
            eps_S = st.number_input("ε_S (Delta)", value=0.2, min_value=1e-6)
        with eps2:
            eps_sigma = st.number_input("ε_σ (Vega)", value=0.002, min_value=1e-6)
        with eps3:
            eps_r = st.number_input("ε_r (Rho)", value=0.0001, min_value=1e-8)

        g = pricer.greeks_finite_diff(p, eps_S=float(eps_S), eps_sigma=float(eps_sigma), eps_r=float(eps_r))

        g1, g2, g3, g4 = st.columns(4)
        g1.metric("Delta", f"{g['delta']:.6f}")
        g2.metric("Vega", f"{g['vega']:.6f}")
        g3.metric("Rho", f"{g['rho']:.6f}")
        g4.metric("Theta (∂V/∂T)", f"{g['theta']:.6f}")

        st.divider()
        st.subheader("Payoff vs moyenne arithmétique avg(S)")

        avg_grid = np.linspace(max(0.0, 0.5 * S), 1.5 * S, 300)
        if option_type == "call":
            payoff = np.maximum(avg_grid - K, 0.0)
            label = "Payoff Call (Asian arith)"
        else:
            payoff = np.maximum(K - avg_grid, 0.0)
            label = "Payoff Put (Asian arith)"

        fig = plt.figure()
        plt.plot(avg_grid, payoff, label=label)
        plt.axvline(K, linestyle="--", label="Strike K")
        plt.title("Payoff vs moyenne arithmétique avg(S)")
        plt.xlabel("avg(S)")
        plt.ylabel("Payoff")
        plt.legend()
        st.pyplot(fig)

# ------------------------------------------------------------
# Branch 2 — Asiatique géométrique (closed-form + MC)
# ------------------------------------------------------------
else:
    st.subheader("Asiatique géométrique — Closed-form + Monte Carlo")
    st.write(
        "Payoff dépend de la **moyenne géométrique** G.\n\n"
        "- Call: max(G - K, 0)\n"
        "- Put : max(K - G, 0)\n"
        "Cette option admet une **formule fermée** (référence) et un pricing Monte Carlo (validation)."
    )

    pricer = AsianGeomMCPricer()

    method = st.selectbox("Méthode", ["Closed-form (référence)", "Monte Carlo", "Comparer les deux"])

    # MC params only if needed
    if method in ["Monte Carlo", "Comparer les deux"]:
        st.subheader("Paramètres Monte Carlo")
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            n_paths = st.select_slider("n_paths", options=[10_000, 20_000, 50_000, 80_000, 120_000], value=80_000)
        with mc2:
            antithetic = st.checkbox("Antithetic variates", value=True)
        with mc3:
            seed = st.number_input("Seed (reproductibilité)", value=42, step=1)
    else:
        n_paths, antithetic, seed = 80_000, True, 42

    st.divider()

    if st.button("Pricer l'asiatique géométrique"):
        cf = None
        mc = None

        if method in ["Closed-form (référence)", "Comparer les deux"]:
            cf = asian_geom_closed_form(float(S), float(K), float(T), float(r), float(sigma), option_type, n_steps=int(n_steps))
            st.subheader("Closed-form (référence)")
            st.metric("Prix closed-form", f"{cf:.6f}")

        if method in ["Monte Carlo", "Comparer les deux"]:
            p = AsianGeomParams(
                S=float(S), K=float(K), T=float(T), r=float(r), sigma=float(sigma),
                option_type=option_type,
                n_steps=int(n_steps),
                n_paths=int(n_paths),
                antithetic=bool(antithetic),
                seed=int(seed),
            )
            mc = pricer.price(p)
            st.subheader("Monte Carlo")
            a, b, c = st.columns(3)
            a.metric("Prix MC", f"{mc.price:.6f}")
            b.metric("Std. error", f"{mc.stderr:.6f}")
            c.metric("IC 95%", f"[{mc.ci_low:.6f} ; {mc.ci_high:.6f}]")
            st.caption(f"Détails: {mc.details}")

        if method == "Comparer les deux" and cf is not None and mc is not None:
            st.divider()
            st.subheader("Comparaison")
            st.metric("Écart (MC - Closed-form)", f"{(mc.price - cf):.6f}")

        st.divider()
        st.subheader("Payoff vs moyenne géométrique G")

        G_grid = np.linspace(0.2 * S, 1.8 * S, 400)
        if option_type == "call":
            payoff = np.maximum(G_grid - K, 0.0)
            label = "Payoff Call (Asian geom)"
        else:
            payoff = np.maximum(K - G_grid, 0.0)
            label = "Payoff Put (Asian geom)"

        fig = plt.figure()
        plt.plot(G_grid, payoff, label=label)
        plt.axvline(K, linestyle="--", label="Strike K")
        plt.title("Payoff vs moyenne géométrique G")
        plt.xlabel("G (moyenne géométrique)")
        plt.ylabel("Payoff")
        plt.legend()
        st.pyplot(fig)
