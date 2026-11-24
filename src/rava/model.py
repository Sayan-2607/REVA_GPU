"""System model (Section III): jobs, servers, settlement rule and effective value (Definition 1, Lemma 1)."""
import numpy as np


def effective_value(v, qhat, kappa):
    """Lemma 1: theta = qhat * v / (qhat + (1 - qhat) * kappa) under pay-on-use with deposit kappa."""
    qh = np.clip(np.asarray(qhat, float), 1e-6, 1.0)
    return qh * np.asarray(v, float) / (qh + (1 - qh) * kappa)


def expected_utility(v, qhat, kappa, price):
    """Equation (3): expected utility of winning at `price`."""
    return qhat * v - price * (qhat + (1 - qhat) * kappa)


def footprint(D, K, L, cap):
    """Equation (2): normalised resource-time footprint D_i = l_i K_i sum_r d_ir / sum_h C_hr."""
    tot = np.asarray(cap, float).sum(0)
    return np.asarray(L) * np.asarray(K) * (np.asarray(D, float) / tot).sum(1)


class Jobs:
    """Arrays describing one epoch. All per-job arrays have length n."""
    def __init__(self, cap, T, D, K, L, A, E, v, q, rho, eta, u, cls, user=None, qhat=None, size=None):
        self.cap = np.ascontiguousarray(cap, float); self.T = int(T)
        self.D = np.ascontiguousarray(D, float); self.K = np.asarray(K, np.int64)
        self.L = np.asarray(L, np.int64); self.A = np.asarray(A, np.int64); self.E = np.asarray(E, np.int64)
        self.v = np.asarray(v, float); self.q = np.asarray(q, float); self.rho = np.asarray(rho, float)
        self.eta = np.asarray(eta, float); self.u = np.asarray(u, float); self.cls = np.asarray(cls, np.int64)
        self.user = np.arange(len(v)) if user is None else np.asarray(user)
        self.qhat = self.q.copy() if qhat is None else np.asarray(qhat, float)
        tot = self.cap.sum(0)
        self.size = (self.L * self.K * (self.D / tot).sum(1)) if size is None else size   # normalised footprint
        win = self.E - self.A + 1
        self.phi = (win - self.L) / win
        self.n = len(v)

    def bids(self, kappa, shade=None):
        qh = np.clip(self.qhat, 1e-6, 1)
        b = qh * self.v / (qh + (1 - qh) * kappa)
        return b if shade is None else b * shade



def jobs_from_records(capacities, records, T, reputation=None, kappa=0.0):
    """Build a `Jobs` instance from plain Python records.

    capacities : list of per-server capacity vectors, e.g. [[8, 64, 256]] (GPU, CPU, GB)
    records    : list of dicts with keys demand (per-server vector), slots, window (a, e), value,
                 and optional gang (K), user, q (true run prob.), qhat (belief), rho, eta, u, cls
    reputation : optional BetaReputation; used for rho when a record has no 'rho'
    """
    n = len(records)
    get = lambda k, d: np.array([r.get(k, d) for r in records], float)
    D = np.array([r["demand"] for r in records], float)
    K = np.array([r.get("gang", 1) for r in records], np.int64)
    L = np.array([r["slots"] for r in records], np.int64)
    A = np.array([r["window"][0] for r in records], np.int64); E = np.array([r["window"][1] for r in records], np.int64)
    if reputation is not None:
        rho = np.array([r.get("rho", reputation.rho(r.get("user", i))) for i, r in enumerate(records)], float)
    else:
        rho = get("rho", 1.0)
    q = np.array([r.get("q", rho[i]) for i, r in enumerate(records)], float)
    qhat = np.array([r.get("qhat", q[i]) for i, r in enumerate(records)], float)
    J = Jobs(np.asarray(capacities, float), T, D, K, L, A, E, get("value", 0.0), q, rho,
             get("eta", 1.0), get("u", 1.0), get("cls", 0).astype(np.int64),
             user=[r.get("user", i) for i, r in enumerate(records)], qhat=qhat)
    return J
