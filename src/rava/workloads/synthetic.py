"""Synthetic workload generator (Section VIII-A). Identical to the generator used for all synthetic results."""
import numpy as np
from ..model import Jobs


def synthetic(n, seed, scale=1.0, bad=0.2, overconf=0.0):
    """Same generator as v1 (one pooled node). scale multiplies cluster capacity (for scalability runs)."""
    rng = np.random.default_rng(seed)
    CAP = np.array([32, 256, 1024], float) * scale; T = 24
    D = np.zeros((n, 3)); L = np.zeros(n, int); A = np.zeros(n, int); E = np.zeros(n, int)
    v = np.zeros(n); q = np.zeros(n); rho = np.zeros(n); eta = np.zeros(n); u = np.zeros(n); cls = np.zeros(n, int)
    for i in range(n):
        c = rng.choice(3, p=[0.35, 0.40, 0.25])
        qi = rng.uniform(0.3, 0.6) if rng.random() < bad else rng.uniform(0.85, 0.99)
        ui = rng.uniform(0.45, 0.98); hist = rng.integers(0, 21); succ = rng.binomial(hist, qi)
        if c == 0: d = [rng.integers(4, 9), rng.integers(16, 65), rng.integers(64, 257)]; Li = rng.integers(4, 13)
        elif c == 1: d = [rng.integers(1, 3), rng.integers(4, 17), rng.integers(16, 65)]; Li = rng.integers(1, 5)
        else: d = [0, rng.integers(32, 129), rng.integers(128, 513)]; Li = rng.integers(2, 9)
        a = rng.integers(0, T - Li + 1); slack = rng.integers(0, T - Li - a + 1)
        Dn = Li * float(np.sum(np.array(d, float) / (CAP / scale)))
        unit = rng.lognormal(0.0, 0.5) * (1.4 if c == 0 else 1.0)
        D[i] = d; L[i] = Li; A[i] = a; E[i] = a + Li - 1 + slack; v[i] = 10.0 * unit * Dn; q[i] = qi
        rho[i] = (4 + succ) / (5 + hist); eta[i] = float(np.clip(ui + rng.normal(0, 0.05), 0.05, 1.0)); u[i] = ui; cls[i] = c
    qhat = np.clip(q + overconf, 0, 1)
    return Jobs(CAP[None, :], T, D, np.ones(n, int), L, A, E, v, q, rho, eta, u, cls, qhat=qhat)

