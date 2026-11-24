"""
RAVA: Reciprocal Adaptive Value and Resource Allocation Auction
Reference implementation + comparative simulation.

Mechanisms compared (all share the same time-slot first-fit placement rule):
  FPA      first-price (pay-as-bid), greedy by bid, shaded bids
  SPA      generalised second-price / critical-value, greedy by bid, truthful
  VCG      exact welfare maximisation (ILP, HiGHS) + Clarke pivot payments
  CA-LOS   Lehmann-O'Callaghan-Shoham greedy b/sqrt(size) + critical payments
  McAfee   multi-unit McAfee trade-reduction double auction (k sellers)
  RAVA     proposed mechanism
  RA-OPT   exact optimum of sum(rho*b) (benchmark allocation only)
"""
import numpy as np, time, json, sys
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import lil_matrix

M_RES = 3                       # GPU, CPU cores, memory GB
CAP = np.array([32, 256, 1024], float)
T = 24                          # hourly slots, one-day epoch
CLASSES = ["train", "infer", "cpu"]

# ------------------------------------------------------------------ workload
def gen_jobs(n, rng, tenants=None):
    jobs = []
    for i in range(n):
        if tenants is None:
            c = rng.choice(3, p=[0.35, 0.40, 0.25])
            p_true = rng.uniform(0.3, 0.6) if rng.random() < 0.2 else rng.uniform(0.85, 0.99)
            u_true = rng.uniform(0.45, 0.98)
            hist = rng.integers(0, 21)
            succ = rng.binomial(hist, p_true)
            rho = (4 + succ) / (5 + hist)               # Beta(4,1) prior, posterior mean
        else:
            tn = tenants[i]
            c, p_true, u_true, rho = tn["c"], tn["p"], tn["u"], tn["a"] / (tn["a"] + tn["b"])
        if c == 0:
            d = [rng.integers(4, 9), rng.integers(16, 65), rng.integers(64, 257)]; L = rng.integers(4, 13)
        elif c == 1:
            d = [rng.integers(1, 3), rng.integers(4, 17), rng.integers(16, 65)]; L = rng.integers(1, 5)
        else:
            d = [0, rng.integers(32, 129), rng.integers(128, 513)]; L = rng.integers(2, 9)
        d = np.array(d, float)
        a = rng.integers(0, T - L + 1)
        slack = rng.integers(0, T - L - a + 1)
        e = a + L - 1 + slack                         # last slot (inclusive)
        D = L * float(np.sum(d / CAP))                # normalised resource-time footprint
        unit = rng.lognormal(0.0, 0.5) * (1.4 if c == 0 else 1.0)
        v = 10.0 * unit * D
        eta = float(np.clip(u_true + rng.normal(0, 0.05), 0.05, 1.0))
        win = e - a + 1
        jobs.append(dict(id=i, c=int(c), d=d, L=int(L), a=int(a), e=int(e), D=D, v=v,
                         p=p_true, u=u_true, rho=rho, eta=eta,
                         phi=(win - L) / win, tenant=(None if tenants is None else i)))
    return jobs

# ------------------------------------------------------------- placement core
def place(order, jobs, keep=None):
    """First-fit in time, in the given priority order. Returns {id: start}."""
    R = np.tile(CAP, (T, 1))
    alloc = {}
    for i in order:
        j = jobs[i]
        for s in range(j["a"], j["e"] - j["L"] + 2):
            if np.all(R[s:s + j["L"]] >= j["d"] - 1e-9):
                R[s:s + j["L"]] -= j["d"]; alloc[i] = s; break
    return alloc

def critical_payments(score, mult, jobs, alloc, eligible):
    """Critical value for a monotone score-ordered greedy (Myerson / LOS)."""
    pay = {}
    for i in alloc:
        others = sorted([k for k in eligible if k != i], key=lambda k: -score[k])
        lo, hi = 0, len(others)          # insert i after others[:pos]
        # largest pos with i still allocated (allocation monotone in pos)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if i in place(others[:mid] + [i], jobs): lo = mid
            else: hi = mid - 1
        crit_score = score[others[lo]] if lo < len(others) else 0.0
        pay[i] = min(crit_score / mult[i], jobs[i]["b"])
    return pay

def greedy_mech(jobs, keyfun, critical=True, pay_as_bid=False):
    ids = list(range(len(jobs)))
    mult = {i: keyfun(jobs[i]) for i in ids}
    score = {i: jobs[i]["b"] * mult[i] for i in ids}
    order = sorted(ids, key=lambda k: -score[k])
    alloc = place(order, jobs)
    if pay_as_bid: pay = {i: jobs[i]["b"] for i in alloc}
    else: pay = critical_payments(score, mult, jobs, alloc, ids)
    return alloc, pay

# --------------------------------------------------------------------- RAVA
RAVA_P = dict(alpha=0.5, gamma=0.0, lam=0.2)
def rava_mult(j, P=RAVA_P, use_rho=True, use_eta=True, use_phi=True, gamma=None, alpha=None):
    P = dict(P)
    if gamma is not None: P["gamma"] = gamma
    if alpha is not None: P["alpha"] = alpha
    m = 1.0 / (j["D"] ** P["alpha"])
    if use_rho: m *= j["rho"]
    if use_eta: m *= j["eta"] ** P["gamma"]
    if use_phi: m *= (1 + P["lam"] * j["phi"])
    return m * j.get("pi", 1.0)

def rava(jobs, **kw):
    return greedy_mech(jobs, lambda j: rava_mult(j, **kw))

# ---------------------------------------------------------------------- ILP
def ilp_opt(jobs, weights, exclude=None, tl=60):
    var = []
    for i, j in enumerate(jobs):
        if i == exclude: continue
        for s in range(j["a"], j["e"] - j["L"] + 2): var.append((i, s))
    nv = len(var)
    c = -np.array([weights[i] for i, _ in var])
    rows = len(jobs) + T * M_RES
    A = lil_matrix((rows, nv))
    for k, (i, s) in enumerate(var):
        A[i, k] = 1
        for t in range(s, s + jobs[i]["L"]):
            for r in range(M_RES):
                if jobs[i]["d"][r] > 0: A[len(jobs) + t * M_RES + r, k] = jobs[i]["d"][r]
    ub = np.concatenate([np.ones(len(jobs)), np.tile(CAP, T)])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), -np.inf, ub), integrality=np.ones(nv),
               bounds=Bounds(0, 1), options=dict(time_limit=tl, mip_rel_gap=1e-6))
    alloc = {var[k][0]: var[k][1] for k in range(nv) if res.x is not None and res.x[k] > 0.5}
    return alloc, -res.fun, res.status == 0

def lp_bound(jobs, weights):
    var = []
    for i, j in enumerate(jobs):
        for s in range(j["a"], j["e"] - j["L"] + 2): var.append((i, s))
    A = lil_matrix((len(jobs) + T * M_RES, len(var)))
    for k, (i, s) in enumerate(var):
        A[i, k] = 1
        for t in range(s, s + jobs[i]["L"]):
            for r in range(M_RES):
                if jobs[i]["d"][r] > 0: A[len(jobs) + t * M_RES + r, k] = jobs[i]["d"][r]
    ub = np.concatenate([np.ones(len(jobs)), np.tile(CAP, T)])
    c = -np.array([weights[i] for i, _ in var])
    res = milp(c, constraints=LinearConstraint(A.tocsr(), -np.inf, ub), integrality=np.zeros(len(var)), bounds=Bounds(0, 1))
    return -res.fun

def vcg(jobs, tl=60):
    w = [j["b"] for j in jobs]
    alloc, W, ok = ilp_opt(jobs, w, tl=tl)
    pay = {}
    for i in alloc:
        _, Wm, ok2 = ilp_opt(jobs, w, exclude=i, tl=tl); ok &= ok2
        pay[i] = max(0.0, Wm - (W - w[i]))
    return alloc, pay, ok

# ------------------------------------------------------------- McAfee DA
def mcafee(jobs, rng, k=8):
    supply_each = M_RES * T / k
    costs = np.sort(rng.uniform(0.2, 0.6, k) * 10.0)        # per normalised unit-hour
    q = np.array([j["b"] / j["D"] for j in jobs]); order = list(np.argsort(-q))
    cum, K, sK = 0.0, 0, 0
    for idx, i in enumerate(order):
        cum += jobs[i]["D"]; s = int(np.ceil(cum / supply_each))
        if s > k or q[i] < costs[s - 1]: break
        K, sK = idx + 1, s
    if K == 0: return {}, {}, 0
    qn = q[order[K]] if K < len(order) else 0.0
    cn = costs[sK] if sK < k else np.inf
    p0 = 0.5 * (qn + cn)
    if costs[sK - 1] <= p0 <= q[order[K - 1]]:
        winners, price = order[:K], p0
    else:                                                    # trade reduction
        winners, price = order[:K - 1], q[order[K - 1]]
    alloc = place(winners, jobs)                             # physical feasibility
    pay = {i: price * jobs[i]["D"] for i in alloc}
    return alloc, pay, len(winners) - len(alloc)

# ------------------------------------------------------------------ metrics
def metrics(jobs, alloc, pay):
    W = sum(jobs[i]["p"] * jobs[i]["v"] for i in alloc)
    rev = sum(pay.values())
    cap = M_RES * T
    used = sum(jobs[i]["p"] * jobs[i]["u"] * jobs[i]["D"] for i in alloc) / cap
    reserved = sum(jobs[i]["D"] for i in alloc) / cap
    acc = []
    for c in range(3):
        n_c = sum(1 for j in jobs if j["c"] == c)
        acc.append(sum(1 for i in alloc if jobs[i]["c"] == c) / max(n_c, 1))
    acc = np.array(acc); jain = acc.sum() ** 2 / (3 * (acc ** 2).sum() + 1e-12)
    return dict(welfare=W, revenue=rev, util=used, reserved=reserved,
                waste=reserved - used, jain=jain, accepted=len(alloc))

def run_one(n, seed, do_vcg=True):
    rng = np.random.default_rng(seed)
    jobs = gen_jobs(n, rng)
    out = {}
    # truthful bids for DSIC mechanisms
    for j in jobs: j["b"] = j["v"]
    t = time.time(); a, p = greedy_mech(jobs, lambda j: 1.0); out["SPA"] = metrics(jobs, a, p); out["SPA"]["t"] = time.time() - t
    t = time.time(); a, p = greedy_mech(jobs, lambda j: 1 / np.sqrt(j["D"])); out["CA-LOS"] = metrics(jobs, a, p); out["CA-LOS"]["t"] = time.time() - t
    t = time.time(); a, p = rava(jobs); out["RAVA"] = metrics(jobs, a, p); out["RAVA"]["t"] = time.time() - t
    a, p = rava(jobs, gamma=1.0); out["RAVA-E"] = metrics(jobs, a, p)
    for name, kw in [("RAVA-no-rho", dict(use_rho=False)), ("RAVA-no-phi", dict(use_phi=False)), ("RAVA-no-size", dict(alpha=0.0))]:
        order = sorted(range(n), key=lambda k: -jobs[k]["b"] * rava_mult(jobs[k], **kw))
        out[name] = metrics(jobs, place(order, jobs), {})
    out["bound"] = lp_bound(jobs, [j["p"] * j["v"] for j in jobs])
    if do_vcg:
        t = time.time(); a, p, ok = vcg(jobs); out["VCG"] = metrics(jobs, a, p); out["VCG"]["t"] = time.time() - t; out["VCG"]["exact"] = ok
        a, _, ok = ilp_opt(jobs, [j["rho"] * j["b"] for j in jobs]); out["RA-OPT"] = metrics(jobs, a, {}); out["RA-OPT"]["exact"] = ok
    t = time.time(); a, p, failed = mcafee(jobs, rng); out["McAfee-DA"] = metrics(jobs, a, p); out["McAfee-DA"]["t"] = time.time() - t; out["McAfee-DA"]["unplaced"] = failed
    # first price: shaded bids (assumption: b = theta * v, theta ~ U(0.6, 0.9))
    for j in jobs: j["b"] = j["v"] * rng.uniform(0.6, 0.9)
    t = time.time(); a, p = greedy_mech(jobs, lambda j: 1.0, pay_as_bid=True); out["FPA"] = metrics(jobs, a, p); out["FPA"]["t"] = time.time() - t
    return out

# ----------------------------------------------------------- multi-epoch
def multi_epoch(seed, epochs=20, n_ten=120, kappa=0.2, bad=0.2):
    rng = np.random.default_rng(seed)
    ten = []
    for _ in range(n_ten):
        ten.append(dict(c=int(rng.choice(3, p=[0.35, 0.4, 0.25])),
                        p=rng.uniform(0.3, 0.6) if rng.random() < bad else rng.uniform(0.85, 0.99),
                        u=rng.uniform(0.45, 0.98), a=4.0, b=1.0))
    res = {"RAVA": [], "CA-LOS": []}
    for ep in range(epochs):
        active = [k for k in range(n_ten) if rng.random() < 0.6]
        jobs = gen_jobs(len(active), rng, tenants=[ten[k] for k in active])
        for j, k in zip(jobs, active): j["b"] = j["v"]; j["tk"] = k
        draws = {k: rng.random() < ten[k]["p"] for k in active}  # common random numbers
        for name in res:
            if name == "RAVA":
                order = sorted(range(len(jobs)), key=lambda q: -jobs[q]["b"] * rava_mult(jobs[q]))
            else:
                order = sorted(range(len(jobs)), key=lambda q: -jobs[q]["b"] / np.sqrt(jobs[q]["D"]))
            alloc = place(order, jobs)
            W = sum(jobs[i]["v"] for i in alloc if draws[jobs[i]["tk"]])
            res[name].append(W)
        # reputation update (Beta posterior) for tenants that RAVA served
        order = sorted(range(len(jobs)), key=lambda q: -jobs[q]["b"] * rava_mult(jobs[q]))
        for i in place(order, jobs):
            k = jobs[i]["tk"]
            if draws[k]: ten[k]["a"] += 1
            else: ten[k]["b"] += 1
    return res

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "main":
        S = int(sys.argv[2]); out = {}
        for n in [40, 80, 120, 160]:
            out[n] = [run_one(n, 1000 * n + s, do_vcg=(n == 40)) for s in range(S)]
            print("done", n, flush=True)
        json.dump(out, open("results_main.json", "w"), default=float)
    elif mode == "sweep":
        out = {}
        for g in [0.0, 0.25, 0.5, 1.0, 2.0]:
            rows = []
            for sd in range(20):
                rng = np.random.default_rng(5000 + sd); jobs = gen_jobs(120, rng)
                for j in jobs: j["b"] = j["v"]
                order = sorted(range(120), key=lambda k: -jobs[k]["b"] * rava_mult(jobs[k], gamma=g))
                m = metrics(jobs, place(order, jobs), {}); rows.append((m["welfare"], m["util"], m["waste"]))
            out[g] = list(np.mean(rows, 0))
        json.dump(out, open("results_sweep.json", "w"), default=float)
    elif mode == "epoch":
        S = int(sys.argv[2])
        bad = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2
        out = [multi_epoch(s, bad=bad) for s in range(S)]
        json.dump(out, open(f"results_epoch_{bad}.json", "w"), default=float)
