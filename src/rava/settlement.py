"""Settlement (Definition 1) and between-epoch adaptation of the reserve price (Section IV-D)."""
import math
import numpy as np


def charge(price, ran, kappa):
    """Pay-on-use: pay `price` if the job ran, forfeit kappa*price on a no-show."""
    return price if ran else kappa * price


def expected_revenue(prices, q, kappa):
    prices = np.asarray(prices, float); q = np.asarray(q, float)
    return float(np.sum(prices * (q + (1 - q) * kappa)))


def update_reserve(psi, utilisation, target, beta=1.0):
    """psi_{k+1} = psi_k * exp(beta * (U_k - U*))."""
    return psi * math.exp(beta * (utilisation - target))


def early_release_credit(price, freed_units, footprint, eps=0.1):
    """Credit eps * p * freed/D for capacity released early and backfilled."""
    return eps * price * freed_units / footprint
