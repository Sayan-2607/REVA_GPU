"""The RAVA mechanism (Section IV): score (7), Algorithm 1 (allocation) and Algorithm 2 (critical prices)."""
import numpy as np
from .placement import place, critical_other


def g_kappa(rho, kappa):
    """Reliability factor matching the settlement rule: g_kappa(rho) = rho + (1 - rho) * kappa."""
    rho = np.asarray(rho, float)
    return rho + (1 - rho) * kappa


def greedy(J, b, w, pay="critical", reserve=0.0):
    """Score-ordered greedy with critical-value payments. score = b * w (w independent of b)."""
    score = b * w
    elig = np.where(b >= reserve * J.size)[0]
    order = elig[np.argsort(-score[elig], kind="stable")].astype(np.int64)
    start, nodes = place(order, J.cap, J.T, J.D, J.K, J.L, J.A, J.E)
    win = np.where(start >= 0)[0]
    p = np.zeros(J.n)
    if pay == "bid":
        p[win] = b[win]
    elif pay == "critical":
        for i in win:
            j = critical_other(order, i, J.cap, J.T, J.D, J.K, J.L, J.A, J.E)
            crit = 0.0 if j < 0 else score[j] / w[i]
            p[i] = min(max(crit, reserve * J.size[i]), b[i])
    return start, nodes, p

def rava_weight(J, kappa, alpha=0.5, gamma=0.0, lam=0.2, pi=None, use_rho=True, rho=None):
    r = J.rho if rho is None else rho
    g = (r + (1 - r) * kappa) if use_rho else 1.0
    w = g * J.eta ** gamma * (1 + lam * J.phi) / J.size ** alpha
    return w if pi is None else w * pi
