import yfinance as yf
import pandas as pd

ticker = yf.Ticker("SPY")
options_dates = ticker.options
print(options_dates)  # Voir les maturités disponibles

# Exemple : on prend la première maturité
opt = ticker.option_chain(options_dates[0])

calls = opt.calls[["strike", "lastPrice", "bid", "ask"]]
#calls.to_csv("options_S&P500call.csv", index=False)
puts = opt.puts[["strike", "lastPrice", "bid", "ask"]]
puts.to_csv("options_S&P500put.csv", index=False)

import pandas as pd
from pathlib import Path

# Chemin vers le dossier data
DATA_DIR = Path(__file__).resolve().parent

def create_rates_csv(filename="rates.csv"):
    data = {
        "maturity": [0.25, 0.5, 1, 2, 5, 10],
        "rate":     [0.038, 0.0375, 0.037, 0.034, 0.028, 0.025]
    }
    df = pd.DataFrame(data)
    filepath = DATA_DIR / filename
    df.to_csv(filepath, index=False)
    print(f"rates.csv créé dans : {filepath}")

if __name__ == "__main__":
    create_rates_csv()
