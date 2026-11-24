"""Placement engine (Section IV-B, IV-C): earliest-fit, best-fit server choice, gang jobs, critical scan.
Compiled with Numba. All mechanisms in this package share this engine, so they differ only in ranking and pricing."""
import numpy as np, numba as nb, math

# ----------------------------------------------------------------------------- placement
@nb.njit(cache=True)
def _cands(R, d, L, s, buf):
    N, T, m = R.shape
    c = 0
    for nd in range(N):
        ok = True
        for t in range(s, s + L):
            for r in range(m):
                if R[nd, t, r] < d[r] - 1e-9:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            buf[c] = nd
            c += 1
    return c

@nb.njit(cache=True)
def try_place(R, d, k, L, a, e, out, apply):
    """Earliest feasible start in [a, e-L+1]; best-fit node choice. Returns start or -1."""
    N = R.shape[0]
    buf = np.empty(N, np.int64)
    for s in range(a, e - L + 2):
        c = _cands(R, d, L, s, buf)
        if c >= k:
            res = np.empty(c)
            for q in range(c):
                nd = buf[q]
                acc = 0.0
                for t in range(s, s + L):
                    acc += R[nd, t, 0] - d[0]
                res[q] = acc
            idx = np.argsort(res)
            for q in range(k):
                nd = buf[idx[q]]
                out[q] = nd
                if apply:
                    for t in range(s, s + L):
                        for r in range(R.shape[2]):
                            R[nd, t, r] -= d[r]
            return s
    return -1

@nb.njit(cache=True)
def init_R(cap, T):
    N, m = cap.shape
    R = np.empty((N, T, m))
    for nd in range(N):
        for t in range(T):
            for r in range(m):
                R[nd, t, r] = cap[nd, r]
    return R

@nb.njit(cache=True)
def place(order, cap, T, D, K, L, A, E):
    R = init_R(cap, T)
    n = D.shape[0]
    start = -np.ones(n, np.int64)
    kmax = K.max()
    nodes = -np.ones((n, kmax), np.int64)
    out = np.empty(kmax, np.int64)
    for i in order:
        s = try_place(R, D[i], K[i], L[i], A[i], E[i], out, True)
        if s >= 0:
            start[i] = s
            for q in range(K[i]):
                nodes[i, q] = out[q]
    return start, nodes

@nb.njit(cache=True)
def critical_other(order, i, cap, T, D, K, L, A, E):
    """Index of the first job j (in `order`, i removed) after whose placement i no longer fits.
    Allocation is monotone in i's position, so i's critical score equals score[j]; -1 if none."""
    R = init_R(cap, T)
    out = np.empty(max(K.max(), 1), np.int64)
    for j in order:
        if j == i:
            continue
        try_place(R, D[j], K[j], L[j], A[j], E[j], out, True)
        if try_place(R, D[i], K[i], L[i], A[i], E[i], out, False) < 0:
            return j
    return -1

