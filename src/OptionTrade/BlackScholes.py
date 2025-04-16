from datetime import datetime

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# Black-Scholes Delta 計算
def bs_delta(s, k, t, r, sigma, option_type):
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    if option_type == "call":
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1


# Black-Scholes 価格計算
def bs_price(s, k, t, r, sigma, option_type):
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    if option_type == "call":
        return s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    else:
        return k * np.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1)


# IV推定（オプション価格から）
def implied_volatility(s, k, t, r, option_price, option_type):
    try:
        # Brent法で σ を逆算
        return brentq(
            lambda sigma: bs_price(s, k, t, r, sigma, option_type) - option_price,
            1e-6,
            5.0,
        )
    except ValueError:
        return np.nan
