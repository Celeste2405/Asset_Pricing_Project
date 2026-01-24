import streamlit as st


def show_page_docs(page_key: str) -> None:
    """
    Affiche un encart 'Procédure' et 'Méthodes' selon la page.

    page_key:
      - "pricing"
      - "calibration"
      - "vol_surface"
      - "asian_geom"
      - "barrier"
    """

    docs = {
       "pricing": {
            "title": " Aide : procédure & méthodes (Pricing)",
            "procedure": """
        1. Choisir le type d’option (Call/Put) et le modèle (**Black–Scholes** ou **Monte Carlo Vanilla**).
        2. Choisir / renseigner la **source des données** :
        - soit **saisie manuelle** (S, σ),
        - soit **données financières extraites** (si disponibles) : S = dernier spot observé, σ = volatilité historique estimée à partir des rendements.
        3. Renseigner : Spot S, Strike K, Maturité T, Taux r (input), Volatilité σ (input ou calculée).
        4. L’application renvoie le prix et le payoff à maturité (graphique).
        5. Si Black–Scholes : greeks analytiques (Delta/Gamma/Vega/Theta).
        6. Si Monte Carlo : prix + erreur standard + IC 95% + greeks par différences finies.
        """,
            "methods": """
        - **Source de données** :
        - Spot **S** : soit saisi, soit récupéré comme dernier prix du sous-jacent (extraction de prix).
        - Volatilité **σ** : soit saisie, soit estimée via **volatilité historique** (écart-type annualisé des rendements log).
        - **Black–Scholes (européennes)** : GBM sous mesure risque-neutre, formule fermée.
        - **Monte Carlo Vanilla (GBM)** : simulation de trajectoires, payoff à maturité, actualisation, moyenne.
        - **Réduction de variance** : antithetic variates (Z et -Z).
        - **Greeks** :
        - analytiques sous BS,
        - numériques sous MC (différences finies : perturbation de S, σ, r, T).
        """,
        },


        "calibration": {
            "title": " Aide : procédure & méthodes (Calibration σ_imp)",
            "procedure": """
        1. Choisir Call/Put et la méthode de calibration (**Brent** ou **Newton**).
        2. Choisir la **source des inputs** :
        - saisie manuelle des paramètres,
        - ou **pré-remplissage depuis la page Pricing** (récupération automatique des valeurs utilisées précédemment).
        3. Entrer S, K, T, r et le **prix de marché** observé :
        - soit saisi manuellement,
        - soit obtenu via un dataset (ex: options Yahoo / fichier CSV si utilisé).
        4. L’application calcule σ_imp tel que BS(S,K,T,r,σ_imp) = Prix_marché.
        5. Contrôle : on recalcule le prix BS avec σ_imp et on affiche l’écart (BS - marché).
        """,
            "methods": """
        - **Objectif** : retrouver la volatilité implicite **σ_imp** cohérente avec un prix observé.
        - **Calibration = inversion** : résoudre f(σ)=BS(σ)-P_mkt=0.
        - **Brent** : solveur robuste (encadrement + convergence stable).
        - **Newton** : solveur rapide utilisant la dérivée (≈ Vega), plus sensible au point initial σ0.
        - **Pré-remplissage (page Pricing → Calibration)** :
        - les paramètres (S, K, T, r, σ) peuvent être stockés dans `st.session_state`
        - la page Calibration peut les réutiliser pour éviter les ressaisies et garantir la cohérence des scénarios.
        """,
        },


        "vol_surface": {
            "title": " Aide : procédure & méthodes (Vol Surface + SABR + Dupire)",
            "procedure": """
1. Choisir la source :
   - A) **Simulée** (démo stable) : on génère des prix marché à partir d’une vol "vraie".
   - B) **Yahoo** (réel) : on charge un tableau d’options filtrées (K, T, price_mkt).
2. Construire la surface σ_imp : pour chaque (K,T), inversion Black–Scholes (Brent/Newton).
3. Visualiser : table des points, matrice σ_imp(K,T), heatmap et (optionnel) surface 3D.
4. (Optionnel) **SABR** : lisser le smile par maturité → σ_SABR(K,T).
5. (Optionnel) **Dupire** : convertir σ(K,T) → prix C(K,T) via BS → calculer σ_loc(K,T) avec les dérivées.
""",
            "methods": """
- **Surface implicite point-par-point** : σ_imp(K,T) via inversion BS sur chaque option.
- **Données Yahoo (qualité)** : filtres (bid/ask valides, strikes autour du spot, volume/OI) pour réduire bruit/illiquidité.
- **SABR (Hagan)** : approximation lognormale, calibration (α,ρ,ν) avec β fixé (stabilité).
- **Dupire (local vol)** :
  - calcul de C(K,T) depuis σ(K,T) via Black–Scholes,
  - dérivées numériques ∂_T C et ∂_{KK} C,
  - σ_loc^2 = (∂_T C) / (0.5 K^2 ∂_{KK} C),
  - garde-fous numériques (clipping / floors) car Dupire est sensible au bruit.
""",
        },

       "asian_options": {
            "title": " Aide : procédure & méthodes (Options Asiatiques)",
            "procedure": """
        1. Choisir le type d’option (Call/Put) et le **type de moyenne** :
        - **Moyenne arithmétique** : payoff dépend de avg(S) (pas de formule fermée simple → Monte Carlo).
        - **Moyenne géométrique** : payoff dépend de G (dispose d’une formule fermée + Monte Carlo).
        2. Entrer S, K, T, r, σ et choisir le nombre de fixings (n_steps).
        3. Selon le choix :
        - **Arithmétique** : choisir les paramètres Monte Carlo (n_paths, antithétiques, seed, option d’inclure S0).
        - **Géométrique** : choisir la méthode (Closed-form / Monte Carlo / Comparer).
        4. Affichage :
        - prix,
        - (si MC) erreur standard + IC 95%,
        - graphiques de payoff (en fonction de avg(S) ou de G),
        - (arithmétique) greeks par différences finies.
        """,
            "methods": """
        - **Asiatique arithmétique (Average Price)** :
        - payoff path-dependent : Call max(avg(S)-K,0), Put max(K-avg(S),0).
        - **Monte Carlo sous GBM** : simulation de trajectoires, calcul de la moyenne arithmétique, payoff, actualisation.
        - **Greeks** : différences finies (seed constant recommandé pour réduire le bruit).
        - **Asiatique géométrique (Geometric Average)** :
        - payoff dépend de G = (∏ S_{t_i})^{1/n}.
        - **Closed-form** : approximation lognormale de G (formule type Black–Scholes sur G) → référence.
        - **Monte Carlo** : validation numérique + convergence vers la formule.
        - **Réduction de variance** : antithetic variates (Z et -Z).
        """,
        },


        "barrier": {
            "title": " Aide : procédure & méthodes (Options Barrière Knock-Out)",
            "procedure": """
1. Choisir Call/Put et le type de barrière :
   - **Up-and-out** : l’option est annulée si S_t ≥ H à un moment.
   - **Down-and-out** : l’option est annulée si S_t ≤ H à un moment.
2. Entrer S, K, H, T, r, σ (et un rebate optionnel).
3. Choisir les paramètres Monte Carlo (n_paths, n_steps, antithétiques, seed).
4. L’application renvoie : prix MC, erreur standard, IC 95% et le **KO rate** (proportion de trajectoires qui touchent la barrière).
""",
            "methods": """
- **Monte Carlo sous GBM** : simulation discrète des trajectoires.
- **Monitoring discret** : la barrière est testée aux dates simulées (biais discret possible).
- **Knock-out** : si la barrière est touchée → payoff = rebate (ici payé à maturité), sinon payoff vanilla.
- **KO rate** : indicateur pédagogique de probabilité de désactivation dans la simulation.
""",
        },
    }

    if page_key not in docs:
        st.warning(f"Docs introuvables pour page_key='{page_key}'.")
        return

    block = docs[page_key]

    with st.expander(block["title"], expanded=False):
        st.markdown("### Procédure")
        st.markdown(block["procedure"])
        st.markdown("### Méthodes utilisées")
        st.markdown(block["methods"])
