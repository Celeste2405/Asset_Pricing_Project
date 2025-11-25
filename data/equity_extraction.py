import yfinance as yf
import pandas as pd

def download_cac40(start="2018-01-01", end="2025-25-11", filename="cac40_prices.csv"):
    data = yf.download("^FCHI", start=start, end=end)
    close = data["Close"]
    close.to_csv(filename)
    print(f"Fichier CAC40 sauvegardé ")

download_cac40(start="2023-01-01", end="2025-11-25", filename="cac40_prices.csv")