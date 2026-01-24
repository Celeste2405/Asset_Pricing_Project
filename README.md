
# 📊 Asset Pricing Project – ENSAI 3A

*Application pédagogique de pricing, calibration et volatilité*

---

## 🎯 Objectif du projet

L’objectif de ce projet est de **concevoir une application pédagogique et modulaire de pricing de produits dérivés**, permettant de couvrir **l’ensemble de la chaîne de valorisation** :

1. **Pricer des options** à partir de paramètres fournis par l’utilisateur,
2. **Calibrer des paramètres de modèle** (volatilité implicite),
3. **Construire, analyser et exploiter une surface de volatilité**,
4. **Comparer plusieurs modèles de pricing** (formules fermées vs Monte Carlo),
5. **Étendre le cadre aux options exotiques** (asiatiques, barrières).

L’application est développée avec **Streamlit** et organisée selon une **séparation claire modèle / test / interface**.

---

## 🧠 Modèles financiers implémentés

### 🔹 Options vanilles

* **Black–Scholes** (Call / Put européens)
* **Monte Carlo Vanilla** (GBM, antithetic variates, IC 95%)
* Greeks :

  * analytiques (Black–Scholes),
  * par différences finies (Monte Carlo)

### 🔹 Calibration

* **Volatilité implicite σ_imp (ω_imp)** :

  * méthode de **Brent** (robuste),
  * méthode de **Newton** (rapide, via Vega).

### 🔹 Volatilité

* **Surface de volatilité implicite σ(K,T)** :

  * données simulées,
  * données réelles (Yahoo Finance).
* **SABR (approximation de Hagan)** pour le lissage du smile.
* **Dupire (volatilité locale)** : passage σ_imp → σ_loc(K,T).

### 🔹 Options exotiques

* **Option asiatique géométrique**

  * formule fermée (référence),
  * Monte Carlo (comparaison et convergence).
* **Options barrières Knock-Out**

  * Up-and-Out / Down-and-Out,
  * pricing Monte Carlo discret,
  * rebate optionnel,
  * taux de knock-out (KO rate).



---

## 🗂️ Structure du projet

```
Asset_Pricing_Project/
│
├── app.py                         # Point d’entrée Streamlit
│
├── pages/                          # Interfaces Streamlit
│   ├── 1_Pricing.py               # Pricing BS & Monte Carlo
│   ├── 2_Calibration.py           # Volatilité implicite
│   ├── 3_Vol_Surface.py           # Surface σ(K,T), SABR, Dupire
│   ├── 4_Asian_Options.py         # Options asiatiques
│   └── 5_Barrier_Options.py       # Options barrières
│
├── model/                          # Logique financière (OOP)
│   ├── equity_model.py            # Black–Scholes, greeks, payoffs
│   ├── mc_vanilla_model.py        # Monte Carlo vanilla
│   ├── calibration_model.py       # σ_imp (Brent / Newton)
│   ├── vol_surface_builder.py     # Construction surface σ(K,T)
│   ├── vol_surface_data.py        # Données simulées / Yahoo
│   ├── sabr_model.py              # SABR (lissage smile)
│   ├── dupire_model.py            # Volatilité locale
│   ├── asian_geom_model.py        # Asiatique géométrique
│   └── barrier_mc_model.py        # Options barrières
│
├── ui/
│   └── page_docs.py               # Documentation intégrée aux pages
│
├── tests/                          # Tests unitaires
│   ├── test_equity_model.py
│   ├── test_calibration_model.py
│   ├── test_vol_surface_model.py
│   ├── test_asian_geom_model.py
│   └── test_barrier_mc_model.py
│
├── requirements.txt
└── README.md
```

---

## 🖥️ Description des interfaces Streamlit

### 1️⃣ **Pricing**

* Choix :

  * Call / Put,
  * Black–Scholes ou Monte Carlo Vanilla.
* Résultats :

  * prix,
  * greeks,
  * payoff à maturité,
  * comparaison BS vs MC (si MC).

---

### 2️⃣ **Calibration – Volatilité implicite**

* Entrée :

  * paramètres de l’option,
  * prix de marché observé,
  * méthode de calibration.
* Sortie :

  * σ_imp,
  * contrôle : écart (BS − marché).

---

### 3️⃣ **Vol Surface**

Deux sources possibles :

#### 🅰️ Données simulées

* Jeu d’options cohérent et stable.
* Idéal pour démonstration pédagogique.

#### 🅱️ Données réelles (Yahoo Finance)

* Extraction automatique via `yfinance`.
* Filtres de qualité :

  * bid/ask valides,
  * moneyness contrôlée,
  * volume / open interest.

**Visualisations :**

* table σ_imp,
* matrice σ(K,T),
* heatmap,
* surface 3D,
* SABR (lissage),
* Dupire (volatilité locale).

---

### 4️⃣ **Options asiatiques (géométriques)**

* Closed-form (référence théorique).
* Monte Carlo (validation numérique).
* Comparaison convergence MC ↔ formule.
* Payoff dépendant de la moyenne géométrique.

---

### 5️⃣ **Options barrières (Knock-Out)**

* Up-and-Out / Down-and-Out.
* Monte Carlo discret.
* Résultats :

  * prix,
  * IC 95%,
  * taux de knock-out,
  * schéma de payoff indicatif.

---

## 🧪 Tests

Chaque module clé est testé indépendamment :

* cohérence prix ↔ σ_imp,
* récupération du σ vrai,
* convergence Monte Carlo,
* comportement limite (barrière très haute / très basse).

Lancement des tests depuis la racine :

```bash
python tests/test_equity_model.py
python tests/test_calibration_model.py
python tests/test_vol_surface_model.py
python tests/test_asian_geom_model.py
python tests/test_barrier_mc_model.py
```

---

## ⚙️ Installation & lancement

### 1️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 2️⃣ Lancer l’application

```bash
python -m streamlit run app.py
```




