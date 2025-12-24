# fichier: tests/test_bs_from_data.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from model.data_utils import load_equity_prices, compute_annual_vol
from model.equity_model import (
    bs_call_price, bs_put_price,
    bs_call_delta, bs_call_gamma, bs_call_vega, bs_call_theta,payoff_call,
    payoff_put
)





ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

def test_payoff_values():
    K = 100

    # Tests simples sur scalaires
    assert payoff_call(80, K) == 0
    assert payoff_call(100, K) == 0
    assert payoff_call(120, K) == 20

    assert payoff_put(80, K) == 20
    assert payoff_put(100, K) == 0
    assert payoff_put(120, K) == 0

    print("✅ Tests scalaires OK")

def test_payoff_plot():
    K = 100
    ST = np.linspace(50, 150, 300)

    c = payoff_call(ST, K)
    p = payoff_put(ST, K)

    # Mini affichage de quelques points
    for x in [80, 100, 120]:
        print(f"ST={x} | Call={payoff_call(x, K)} | Put={payoff_put(x, K)}")

    # Plot
    plt.figure()
    plt.plot(ST, c, label="Payoff Call")
    plt.plot(ST, p, label="Payoff Put")
    plt.axvline(K, linestyle="--", label="Strike K")
    plt.title("Payoff à maturité")
    plt.xlabel("S_T")
    plt.ylabel("Payoff")
    plt.legend()
    plt.show()



if __name__ == "__main__":

    # 1. Charger les prix et calculer S et sigma
    prices = load_equity_prices()
    ticker = "^FCHI"   # à adapter à ce que tu as téléchargé
    S = prices[ticker].dropna().iloc[-1]   # dernier prix
    sigma = compute_annual_vol(prices, ticker)


    T = 1.0  # maturité 1 an
    # 3. Fixer un strike (par exemple at-the-money)
    # option at-the-money
    K = S 
    r = 0.025   # input choisi par l’utilisateur
    call = bs_call_price(S, K, T, r, sigma)



    # 4. Calculer prix et grecques
    call_price = bs_call_price(S, K, T, r, sigma)
    put_price  = bs_put_price(S, K, T, r, sigma)
    delta = bs_call_delta(S, K, T, r, sigma)
    gamma = bs_call_gamma(S, K, T, r, sigma)
    vega  = bs_call_vega(S, K, T, r, sigma)
    theta = bs_call_theta(S, K, T, r, sigma)

    print(f"Sous-jacent : {ticker}")
    print(f"S = {S:.2f}, K = {K:.2f}, T = {T}, r = {r:.4f}, sigma = {sigma:.4f}")
    print(f"Prix call BS : {call_price:.4f}")
    print(f"Prix put  BS : {put_price:.4f}")
    print(f"Delta : {delta:.4f}, Gamma : {gamma:.6f}, Vega : {vega:.4f}, Theta : {theta:.4f}")
    test_payoff_values()
    test_payoff_plot()

