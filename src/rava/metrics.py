"""Evaluation metrics (Section VIII-A)."""
import numpy as np


def metrics(J, start, p, kappa, W_override=None):
    win = start >= 0
    W = float(np.sum(J.q[win] * J.v[win])) if W_override is None else W_override
    rev = float(np.sum(p[win] * (J.q[win] + (1 - J.q[win]) * kappa)))
    cap = J.cap.sum(0); m = J.cap.shape[1]
    foot = J.L * J.K * (J.D / cap).sum(1) / (m * J.T)                     # share of the epoch's capacity
    gpu = J.L * J.K * J.D[:, 0] / (cap[0] * J.T)
    acc = np.array([win[J.cls == c].mean() if np.any(J.cls == c) else np.nan for c in np.unique(J.cls)])
    acc = acc[~np.isnan(acc)]
    return dict(welfare=W, revenue=rev, served=int(win.sum()),
                util=float(np.sum((J.q * J.u * foot)[win])), reserved=float(np.sum(foot[win])),
                gpu_used=float(np.sum((J.q * J.u * gpu)[win])), gpu_lost=float(np.sum(((1 - J.q) * gpu)[win])),
                gpu_idle=float(np.sum((J.q * (1 - J.u) * gpu)[win])),
                jain=float(acc.sum() ** 2 / (len(acc) * (acc ** 2).sum() + 1e-12)))

