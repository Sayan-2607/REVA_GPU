"""LP-relaxation upper bound on expected welfare (pooled capacity): used to report efficiency."""
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix


def lp_bound(J, weights):
    """LP relaxation of max sum w_i x_i with pooled capacity (node constraints relaxed): an upper bound."""
    tot = J.cap.sum(0); rows, cols, vals = [], [], []; c = []; nv = 0
    m = J.cap.shape[1]
    for i in range(J.n):
        for s in range(J.A[i], J.E[i] - J.L[i] + 2):
            c.append(-weights[i]); rows.append(i); cols.append(nv); vals.append(1.0)
            for t in range(s, s + J.L[i]):
                for r in range(m):
                    if J.D[i, r] > 0:
                        rows.append(J.n + t * m + r); cols.append(nv); vals.append(J.D[i, r] * J.K[i])
            nv += 1
    A = coo_matrix((vals, (rows, cols)), shape=(J.n + J.T * m, nv)).tocsr()
    ub = np.concatenate([np.ones(J.n), np.tile(tot, J.T)])
    res = milp(np.array(c), constraints=LinearConstraint(A, -np.inf, ub), integrality=np.zeros(nv), bounds=Bounds(0, 1))
    return -res.fun
