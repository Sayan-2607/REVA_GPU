"""Closed-form results of Section V as functions (used by tests and the CLI)."""
import math


def g_kappa(rho, kappa):
    return rho + (1 - rho) * kappa


def misspecification(q, qhat, kappa):
    """Proposition 1, Eq. (8): factor by which the RAVA score over-states expected value (rho = q)."""
    return qhat * (q + (1 - q) * kappa) / (q * (qhat + (1 - qhat) * kappa))


def misspecification_naive(qhat, kappa):
    """Proposition 1(3): factor for the naive score g(rho) = rho."""
    return qhat / (qhat + (1 - qhat) * kappa)


def lipschitz(kappa, vbar, rho_min):
    """Lemma 3: Lambda = (1 - kappa) * Vbar / g_kappa(rho_min)."""
    return (1 - kappa) * vbar / g_kappa(rho_min, kappa)


def epsilon_bound(delta, kappa, vbar, rho_min, mass, a0=4.0, b0=1.0, zeta=1.0):
    """Theorem 2, Eq. (9): largest discounted gain from deviating in one epoch."""
    lam = lipschitz(kappa, vbar, rho_min)
    if zeta >= 1.0:
        return delta * lam / ((1 - delta) * mass)
    return delta * lam / ((1 - delta * zeta) * (a0 + b0))


def defaults_before_threshold_no_forgetting(s, f, rho_bar, a0=4.0, b0=1.0):
    """Proposition 4 (zeta = 1): defaults an exploiter can make while keeping rho >= rho_bar."""
    return max(0, math.floor((a0 + s) / rho_bar - (a0 + b0 + s + f)))


def rho_after_defaults_bound(m, a0=4.0, b0=1.0, zeta=0.9):
    """Proposition 4 (zeta < 1): upper bound on rho after m consecutive defaults, for any history."""
    return (a0 + zeta ** m / (1 - zeta)) / (a0 + b0 + (1 - zeta ** m) / (1 - zeta))


def max_defaults_with_forgetting(rho_bar, a0=4.0, b0=1.0, zeta=0.9, cap=10_000):
    """Proposition 4: smallest m with the bound below rho_bar (history-independent constant m*)."""
    limit = a0 / (a0 + b0 + 1 / (1 - zeta))
    if rho_bar <= limit:
        return math.inf
    for m in range(cap):
        if rho_after_defaults_bound(m, a0, b0, zeta) < rho_bar:
            return m
    return math.inf
