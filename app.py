import streamlit as st

st.set_page_config(
    page_title="Pricing de produits dérivés",
    layout="wide"
)

st.title("🧮 Pricing de produits dérivés")
st.subheader("Application pédagogique de pricing et calibration de produits dérivés")

st.write(
    """
Bienvenue dans le **Pricing de produits dérivés**, une application Streamlit développée dans le cadre du
cours de **Modélisation et évaluation des produits dérivés (ENSAI 3A)**.

Cette application couvre **toute la chaîne de valorisation des options**, depuis le pricing
jusqu’à la construction de surfaces de volatilité et le traitement d’options exotiques.
"""
)

st.divider()

st.markdown(
    """
### 📌 Fonctionnalités principales

L’application est organisée en **pages indépendantes**, accessibles via le menu latéral :

#### 1️⃣ Pricing
- Pricing d’options **européennes Call / Put**
- Modèles :
  - **Black–Scholes** (formule fermée)
  - **Monte Carlo Vanilla** (GBM, IC 95%)
- Résultats :
  - prix,
  - greeks (analytiques ou numériques),
  - payoff à maturité,
  - comparaison BS vs MC.

#### 2️⃣ Calibration – Volatilité implicite
- Inversion de Black–Scholes à partir d’un **prix de marché**
- Méthodes :
  - **Brent** (robuste),
  - **Newton** (rapide, via Vega)
- Sortie :
  - **σ_imp (ω_imp)**,
  - contrôle de cohérence (BS − marché).

#### 3️⃣ Vol Surface
- Construction de la surface **σ_imp(K,T)**
- Sources :
  - données **simulées** (démonstration stable),
  - données **réelles Yahoo Finance**
- Visualisations :
  - table,
  - heatmap,
  - surface 3D
- Extensions :
  - **SABR** (lissage du smile),
  - **Dupire** (volatilité locale).

#### 4️⃣ Options Asiatiques
- **Moyenne arithmétique** (Monte Carlo)
- **Moyenne géométrique** :
  - formule fermée (référence),
  - Monte Carlo,
  - comparaison MC vs closed-form.

#### 5️⃣ Options Barrières (Knock-Out)
- Up-and-Out / Down-and-Out
- Pricing **Monte Carlo discret**
- Rebate optionnel
- Indicateur de **knock-out rate**
"""
)
 
st.divider()

st.info(
    "👉 Utilisez le **menu à gauche** pour naviguer entre les pages.\n\n"
    "Chaque page contient un encart *Procédure & Méthodes* expliquant les modèles utilisés."
)

st.caption(
    "Projet ENSAI 3A — Modélisation financière • Produits dérivés • Streamlit • Python"
)
