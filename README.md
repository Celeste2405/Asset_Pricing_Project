# Asset_Pricing_Project

# Objectif du projet

# 📘 Asset Pricing – Projet ENSAI 2025–2026

## 🎯 Objectif du projet

Construire, en Python, une **mini-application de pricing** avec interface graphique (Streamlit / Dash)
qui reproduit les grandes étapes d’un projet de finance de marché :

1. **Extraction de données de marché** : courbe de taux, marchés actions, volatilité implicite.
2. **Pricing d’instruments de taux “vanille” (sans modèle dynamique)** :
   - un **bond avec coupons**,
   - et **un swap OU un future de taux**.
3. **Implémentation et calibration d’un modèle equity Black–Scholes**.
4. **Pricing d’un produit optionnel equity avec grecques** (delta, gamma, vega, theta).
5. **Visualisation et vulgarisation** via une mini-interface graphique pour un utilisateur non technique.

Ce projet suit exactement les étapes demandées dans le sujet de l’UE Asset Pricing.

---

## 🧩 Structure du projet

```text
projet_asset_pricing/
│
├── app.py                  # Application principale (Streamlit / Dash)
│
├── data/                   # Données de marché (taux, actions, options)
│   ├── rates.csv           # Courbe de taux (maturité, taux)
│   ├── equity_prices.csv   # Prix historiques des actions / indices
│   └── options.csv         # (optionnel) Prix d'options de marché pour calibration
│
├── models/
│   ├── data_extraction.py  # Extraction, nettoyage et statistiques de base
│   ├── rates_pricing.py    # Pricing bond + swap/future de taux
│   ├── equity_model.py     # Modèle Black–Scholes + calibration
│   └── derivatives_pricing.py  # Produit optionnel equity + grecques
│
└── README.md               # Ce fichier

poids du portefeuille optimal
