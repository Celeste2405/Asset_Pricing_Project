# Asset_Pricing_Project

# Objectif du projet

Construire une mini-application Streamlit qui montre comment :

on valorise une option européenne avec le modèle de Black–Scholes (prix + grecques),

on construit une frontière efficiente de Markowitz pour optimiser un portefeuille d’actions,

en utilisant des données de marché réelles ou simulées.

Tout doit être présenté dans une interface simple, manipulable par un utilisateur non spécialiste.

# 1. Structure globale du projet

Le projet sera organisé comme ceci :

projet_asset_pricing/
│
├── app.py                # Application Streamlit (interface principale)
├── data/                 # Données (historique actions, courbe de taux…)
├── models/               # Toute la logique métier
│   ├── black_scholes.py  # Pricing d’options + grecques
│   ├── markowitz.py      # Optimisation de portefeuille
│   └── utils.py          # Fonctions génériques (chargement CSV, stats…)
│
└── README.md             # Explication du projet (ce fichier)

# 2. CONTENU DU CODE
   
2.1 Données (data/)

Nous aurons besoin :

d’un historique de prix d’actions (via Yahoo Finance ou CSV),

d’un taux sans risque simple (constant ou issu d’une courbe de taux),

d’une estimation de :

rendements moyens,

volatilités,

corrélations,
basées sur les rendements historiques.
Ces données seront utilisées à la fois pour Black–Scholes et pour Markowitz.

2.2 Module Black–Scholes (models/black_scholes.py)

On implémentera :

✔ Prix du call européen
call_price_bs(S, K, T, r, sigma)
Grecques principales

delta

gamma

vega

theta

Ces fonctions seront appelées depuis l’interface Streamlit pour afficher :

le prix de l’option,

comment il réagit aux changements des paramètres.

2.3 Module Markowitz (models/markowitz.py)

On implémentera :

✔ Calcul du rendement ↗ et du risque ↔ d’un portefeuille donné

(somme pondérée des rendements + variance/covariance)

✔ Construction de la frontière efficiente

Pour plusieurs objectifs de rendement (ou de risque), on calcule :

le portefeuille de variance minimale,

et on affiche la courbe rendement/risque.

2.4 Interface Streamlit (app.py)

L’app comportera 3 pages :

🟦 Page 1 – Données

Sélectionner une action

Visualiser :

prix historiques,

rendements,

volatilité

Montrer les statistiques de base utilisées dans les modules.

🟥 Page 2 – Pricing d’une option européenne

Saisie des paramètres (S, K, T, taux, volatilité)

Calcul instantané :

prix du call Black–Scholes

delta, gamma, vega, theta

Graphiques interactifs (par exemple prix en fonction de S ou σ)

🟩 Page 3 – Portefeuille de Markowitz

Sélection d’un panier d’actions

Construction de la frontière efficiente

Visualisation :

courbe rendement/risque

poids du portefeuille optimal
