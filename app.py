import streamlit as st

st.set_page_config(page_title="Asset Pricing – ENSAI", layout="wide")

st.title("Asset Pricing – Application Streamlit")
st.write(
    """
Cette application contient **3 interfaces indépendantes** :

1. **Pricing** : l'utilisateur choisit un modèle + paramètres → prix + greeks + payoff
2. **Calibration** : l'utilisateur choisit une méthode + paramètres + prix marché → **σ_imp (omega imp)**
3. **Vol Surface** : construction et visualisation de **σ_imp(K, T)**
"""
)

st.info("⚠️ Comme demandé : le taux r est un **input utilisateur** (jamais imposé automatiquement).")
st.write("Utilise le menu à gauche (pages) pour naviguer.")

