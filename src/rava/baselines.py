"""Baseline mechanisms (Section VIII-A): first-price, second-price, LOS, G-VMPAC-II, online primal-dual (OPD),
DRF, overbooking, McAfee double auction and exact VCG."""
import numpy as np, numba as nb, math
from .placement import _cands, try_place, init_R, place
from .mechanism import greedy

def opd(J, b, Lp, Up):
    """Online primal-dual posted-price auction (Zhang et al. style): jobs in arrival order, exponential
    price per (node, slot, resource) unit, accept iff bid >= cheapest feasible bundle price."""
    order = np.argsort(J.A, kind="stable").astype(np.int64)
    start, nodes, p = _opd(order, J.cap, J.T, J.D, J.K, J.L, J.A, J.E, b, float(Lp), float(Up))
    return start, nodes, p

@nb.njit(cache=True)
def _opd(order, cap, T, D, K, L, A, E, b, Lp, Up):
    N, m = cap.shape
    R = init_R(cap, T)
    n = D.shape[0]
    start = -np.ones(n, np.int64); kmax = K.max(); nodes = -np.ones((n, kmax), np.int64); p = np.zeros(n)
    tot = np.zeros(m)
    for r in range(m):
        for nd in range(N):
            tot[r] += cap[nd, r]
    base = 2.0 * math.e * Up / Lp
    buf = np.empty(N, np.int64)
    for i in order:
        best = 1e300; bs = -1; bn = np.empty(K[i], np.int64)
        for s in range(A[i], E[i] - L[i] + 2):
            c = _cands(R, D[i], L[i], s, buf)
            if c < K[i]:
                continue
            cost = np.empty(c)
            for q in range(c):
                nd = buf[q]; acc = 0.0
                for t in range(s, s + L[i]):
                    for r in range(m):
                        if cap[nd, r] > 0 and D[i, r] > 0:
                            y = 1.0 - R[nd, t, r] / cap[nd, r]
                            acc += D[i, r] / tot[r] * (Lp / (2 * math.e)) * base ** y
                cost[q] = acc
            idx = np.argsort(cost)
            tc = 0.0
            for q in range(K[i]):
                tc += cost[idx[q]]
            if tc < best:
                best = tc; bs = s
                for q in range(K[i]):
                    bn[q] = buf[idx[q]]
        if bs >= 0 and b[i] >= best:
            start[i] = bs; p[i] = best
            for q in range(K[i]):
                nd = bn[q]; nodes[i, q] = nd
                for t in range(bs, bs + L[i]):
                    for r in range(m):
                        R[nd, t, r] -= D[i, r]
    return start, nodes, p

def drf(J):
    """Price-free Dominant Resource Fairness: repeatedly serve the user with the smallest dominant share."""
    import heapq
    R = init_R(J.cap, J.T); tot = J.cap.sum(0) * J.T
    users = {}
    for i in np.argsort(J.A, kind="stable"):
        users.setdefault(J.user[i], []).append(i)
    share = {u: 0.0 for u in users}; heap = [(0.0, k, u) for k, u in enumerate(users)]; heapq.heapify(heap)
    start = -np.ones(J.n, np.int64); out = np.empty(max(J.K.max(), 1), np.int64)
    ptr = {u: 0 for u in users}
    while heap:
        sh, k, u = heapq.heappop(heap)
        if ptr[u] >= len(users[u]): continue
        i = users[u][ptr[u]]; ptr[u] += 1
        s = try_place(R, J.D[i], J.K[i], J.L[i], J.A[i], J.E[i], out, True)
        if s >= 0:
            start[i] = s; share[u] += float(np.max(J.D[i] * J.K[i] * J.L[i] / tot))
        if ptr[u] < len(users[u]): heapq.heappush(heap, (share[u], k, u))
    return start, None, np.zeros(J.n)

def overbook(J, b, tau=0.1, samples=30, seed=0):
    """LOS ranking on capacity inflated by (1+tau); on a shortage at run time the lowest-ranked shown jobs are
    evicted (Oh-Trust-style overbooking). Returns start and the expected value of non-evicted shown jobs."""
    w = 1 / np.sqrt(J.size); score = b * w
    order = np.argsort(-score, kind="stable").astype(np.int64)
    start, nodes = place(order, J.cap * (1 + tau), J.T, J.D, J.K, J.L, J.A, J.E)
    win = [i for i in order if start[i] >= 0]
    rng = np.random.default_rng(seed); W = 0.0; ev = 0.0
    for _ in range(samples):
        shown = [i for i in win if rng.random() < J.q[i]]
        load = np.zeros((J.cap.shape[0], J.T, J.cap.shape[1]))
        for i in shown:
            for nd in nodes[i, :J.K[i]]: load[nd, start[i]:start[i] + J.L[i]] += J.D[i]
        kept = set(shown)
        for i in reversed(shown):
            over = False
            for nd in nodes[i, :J.K[i]]:
                if np.any(load[nd, start[i]:start[i] + J.L[i]] > J.cap[nd] + 1e-9): over = True
            if over:
                kept.discard(i); ev += 1
                for nd in nodes[i, :J.K[i]]: load[nd, start[i]:start[i] + J.L[i]] -= J.D[i]
        W += sum(J.v[i] for i in kept)
    return start, W / samples, ev / samples

def mcafee(J, b, rng, k=8):
    """Multi-unit McAfee trade reduction on normalised resource-time units, then physical placement."""
    supply = J.size.sum() * 0 + J.cap.shape[1] * J.T / k * (1.0)   # each seller owns 1/k of every resource-slot
    costs = np.sort(rng.uniform(0.2, 0.6, k) * np.median(b / J.size))
    qv = b / J.size; order = list(np.argsort(-qv)); cum = 0.0; Kb = 0; sK = 0
    for idx, i in enumerate(order):
        cum += J.size[i]; s = int(np.ceil(cum / supply))
        if s > k or qv[i] < costs[s - 1]: break
        Kb, sK = idx + 1, s
    start = -np.ones(J.n, np.int64); p = np.zeros(J.n)
    if Kb == 0: return start, None, p, 0
    qn = qv[order[Kb]] if Kb < len(order) else 0.0; cn = costs[sK] if sK < k else np.inf; p0 = 0.5 * (qn + cn)
    if costs[sK - 1] <= p0 <= qv[order[Kb - 1]]: winners, price = order[:Kb], p0
    else: winners, price = order[:Kb - 1], qv[order[Kb - 1]]
    st, nodes = place(np.array(winners, np.int64), J.cap, J.T, J.D, J.K, J.L, J.A, J.E)
    for i in winners:
        if st[i] >= 0: start[i] = st[i]; p[i] = price * J.size[i]
    return start, nodes, p, int(len(winners) - (st[winners] >= 0).sum())



# ----------------------------------------------------------------------------- simple greedy baselines
def first_price(J, b, shade):
    """Pay-as-bid greedy by (shaded) bid."""
    return greedy(J, b * shade, np.ones(J.n), pay="bid")


def second_price(J, b):
    """Greedy by raw bid with critical-value prices (generalised second price)."""
    return greedy(J, b, np.ones(J.n))


def los(J, b):
    """Lehmann-O'Callaghan-Shoham: greedy by b / sqrt(size) with critical prices."""
    return greedy(J, b, 1 / np.sqrt(J.size))


def gvmpac2(J, b):
    """G-VMPAC-II style density greedy (relevance factor 1/C_r): b / size, critical prices."""
    return greedy(J, b, 1 / J.size)


# ----------------------------------------------------------------------------- exact VCG (one pooled server)
def _wdp(J, weights, exclude=-1, time_limit=60):
    """Time-indexed winner-determination ILP on a single pooled server (HiGHS)."""
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    if J.cap.shape[0] != 1 or np.any(J.K != 1):
        raise NotImplementedError("exact VCG is implemented for one pooled server and K_i = 1")
    m = J.cap.shape[1]; var = []
    for i in range(J.n):
        if i == exclude: continue
        for s in range(J.A[i], J.E[i] - J.L[i] + 2): var.append((i, s))
    A = lil_matrix((J.n + J.T * m, len(var)))
    for k, (i, s) in enumerate(var):
        A[i, k] = 1
        for t in range(s, s + J.L[i]):
            for r in range(m):
                if J.D[i, r] > 0: A[J.n + t * m + r, k] = J.D[i, r]
    ub = np.concatenate([np.ones(J.n), np.tile(J.cap[0], J.T)])
    res = milp(-np.array([weights[i] for i, _ in var]), constraints=LinearConstraint(A.tocsr(), -np.inf, ub),
               integrality=np.ones(len(var)), bounds=Bounds(0, 1), options=dict(time_limit=time_limit, mip_rel_gap=1e-6))
    start = -np.ones(J.n, np.int64)
    if res.x is not None:
        for k, (i, s) in enumerate(var):
            if res.x[k] > 0.5: start[i] = s
    return start, (-res.fun if res.x is not None else 0.0), res.status == 0


def vcg(J, b, time_limit=60):
    """Exact VCG with Clarke pivot payments. Returns start, payments, and whether every ILP was proven optimal."""
    start, W, ok = _wdp(J, b, time_limit=time_limit); p = np.zeros(J.n)
    for i in np.where(start >= 0)[0]:
        _, Wm, ok2 = _wdp(J, b, exclude=i, time_limit=time_limit); ok &= ok2
        p[i] = max(0.0, Wm - (W - b[i]))
    return start, p, ok
