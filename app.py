import streamlit as st

st.set_page_config(
    page_title="Pricing de produits dérivés",
    layout="wide"
)

st.title("Pricing de produits dérivés")
st.subheader("Application pédagogique de pricing et calibration de produits dérivés réalisée par Céleste NENEHIDINI, Mariane ALAPINI et Ikram DRINE")

st.write(
    """
Bienvenue sur notre application développée dans le cadre du
cours de **Modélisation et évaluation des produits dérivés**.

Cette application couvre **toute la chaîne de valorisation des options**, depuis le pricing
jusqu’à la construction de surfaces de volatilité ainsi que le traitement d’options exotiques.
"""
)

st.divider()

st.markdown(
    """
### Les fonctionnalités principales

L’application est organisée en **pages indépendantes**, accessibles via le menu latéral :

#### Page 1 : Pricing
- Pricing d’options **européennes Call / Put**
- Modèles :
  - **Black–Scholes** (avec la formule fermée)
  - **Monte Carlo Vanilla** (par GBM, et calcul des IC à 95%)
- Résultats :
  - prix,
  - greeks (analytiques ou numériques),
  - payoff à maturité,
  - comparaison Black-Scholes vs Monte-Carlo.

#### Page 2 : Calibration – Volatilité implicite
- Inversion de Black–Scholes à partir d’un **prix de marché**
- Méthodes :
  - **Brent**,
  - **Newton**
- Sortie :
  - **σ_imp**,
  - contrôle de cohérence (BS − marché).

#### Page 3 : Vol Surface
- Construction de la surface **σ_imp(K,T)**
- Sources :
  - données **simulées**,
  - données **réelles Yahoo Finance**
- Visualisations :
  - table,
  - heatmap,
  - surface 3D
- Extensions :
  - **SABR** (lissage du smile),
  - **Dupire** (volatilité locale).

#### Page 4 : Options Asiatiques
- **Moyenne arithmétique** (Monte Carlo)
- **Moyenne géométrique** :
  - formule fermée,
  - Monte Carlo,
  - comparaison MC vs closed-form.

#### Page 5 : Options Barrières (Knock-Out)
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
