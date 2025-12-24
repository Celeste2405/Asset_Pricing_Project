# fichier: models/data_utils.py (par exemple)
import pandas as pd
import numpy as np
from pathlib import Path



def load_equity_prices():
    filepath = r"c:\Users\werid\Desktop\Cours 3A\modéle d'evaluation de produits dérivées\Asset pricing\Asset_Pricing_Project\data\cac40_prices.csv"

    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df

def compute_annual_vol(prices, ticker, trading_days=252):
    """
    prices : DataFrame de prix (colonnes = tickers)
    ticker : ticker à utiliser (ex: 'AAPL')
    """
    series = prices[ticker].dropna()
    # rendements log ou simples, ici log :
    log_returns = np.log(series / series.shift(1)).dropna()
    daily_vol = log_returns.std()
    annual_vol = daily_vol * np.sqrt(trading_days)
    return annual_vol


prices = load_equity_prices()
sigma = compute_annual_vol(prices, "^FCHI")
print("Vol annualisée ^FCHI  :", sigma)
