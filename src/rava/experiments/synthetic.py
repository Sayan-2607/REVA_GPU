"""Synthetic experiments for RAVA v2. Usage: python3 experiments.py <name> [args]"""
import sys, json, time, numpy as np
from rava import core as C

def run_all_mechs(J, kappa=0.0, seed=0, with_bound=True):
    b = J.bids(kappa); out = {}
    los = 1 / np.sqrt(J.size)
    for k, w in [("SPA", np.ones(J.n)), ("LOS", los), ("GVMPAC2", 1 / J.size),
                 ("RAVA", C.rava_weight(J, kappa)), ("RAVA-E", C.rava_weight(J, kappa, gamma=1.0)),
                 ("RAVA-a1", C.rava_weight(J, kappa, alpha=1.0)), ("RAVA-no-rho", C.rava_weight(J, kappa, use_rho=False)),
                 ("RAVA-no-phi", C.rava_weight(J, kappa, lam=0.0)), ("RAVA-no-size", C.rava_weight(J, kappa, alpha=0.0))]:
        t = time.time(); st, nd, p = C.greedy(J, b, w); out[k] = C.metrics(J, st, p, kappa); out[k]["t"] = time.time() - t
    rng = np.random.default_rng(seed + 11)
    st, nd, p = C.greedy(J, b * rng.uniform(0.6, 0.9, J.n), np.ones(J.n), pay="bid"); out["FPA"] = C.metrics(J, st, p, kappa)
    dens = b / J.size; t = time.time()
    st, nd, p = C.opd(J, b, np.percentile(dens, 5), np.percentile(dens, 95)); out["OPD"] = C.metrics(J, st, p, kappa); out["OPD"]["t"] = time.time() - t
    st, _, p = C.drf(J); out["DRF"] = C.metrics(J, st, p, kappa)
    st, _, p, unp = C.mcafee(J, b, rng); out["McAfee"] = C.metrics(J, st, p, kappa); out["McAfee"]["unplaced"] = unp
    st, W, ev = C.overbook(J, b, 0.1, 30, seed); out["Overbook"] = C.metrics(J, st, np.zeros(J.n), kappa, W_override=W); out["Overbook"]["evicted"] = ev
    if with_bound: out["bound"] = C.lp_bound(J, J.q * J.v)
    return out

def main_exp(seeds=50):
    res = {}
    for n in [40, 80, 120, 160]:
        res[n] = [run_all_mechs(C.synthetic(n, 1000 * n + s), 0.0, s) for s in range(seeds)]
        print("main", n, flush=True)
    json.dump(res, open("res_main.json", "w"))

def kappa_exp(seeds=50):
    res = []
    for belief, oc in [("calibrated", 0.0), ("overconfident+0.3", 0.3), ("certain", 1.0)]:
        for kappa in [0.0, 0.25, 0.5, 1.0]:
            for s in range(seeds):
                J = C.synthetic(120, 120000 + s, overconf=oc); b = J.bids(kappa)
                row = dict(belief=belief, kappa=kappa, seed=s)
                for k, w in [("LOS", 1 / np.sqrt(J.size)), ("RAVA", C.rava_weight(J, kappa)),
                             ("RAVA-naive", C.rava_weight(J, 0.0))]:          # naive: g = rho regardless of kappa
                    st, nd, p = C.greedy(J, b, w, pay="none"); row[k] = C.metrics(J, st, p, kappa)["welfare"]
                res.append(row)
        print("kappa", belief, flush=True)
    json.dump(res, open("res_kappa.json", "w"))

def bad_exp(seeds=50):
    res = []
    for bad in [0.0, 0.2, 0.4, 0.6]:
        for s in range(seeds):
            J = C.synthetic(120, 120000 + s, bad=bad); b = J.bids(0.0); row = dict(bad=bad, seed=s)
            for k, w in [("LOS", 1 / np.sqrt(J.size)), ("GVMPAC2", 1 / J.size), ("RAVA", C.rava_weight(J, 0.0))]:
                st, nd, p = C.greedy(J, b, w, pay="none"); row[k] = C.metrics(J, st, p, 0.0)["welfare"]
            st, W, ev = C.overbook(J, b, 0.1, 30, s); row["Overbook"] = W
            row["bound"] = C.lp_bound(J, J.q * J.v); res.append(row)
        print("bad", bad, flush=True)
    json.dump(res, open("res_bad.json", "w"))

# ------------------------------------------------------------------ multi-round population model
def population(rng, n_ten, bad=0.2):
    return [dict(c=int(rng.choice(3, p=[0.35, 0.4, 0.25])),
                 q=rng.uniform(0.3, 0.6) if rng.random() < bad else rng.uniform(0.85, 0.99),
                 u=rng.uniform(0.45, 0.98), s=0.0, f=0.0, kind="normal") for _ in range(n_ten)]

def epoch_jobs(rng, tens, ids, a0, b0, rho_min):
    """Fresh job per active tenant; rho from the (possibly discounted) Beta counts."""
    n = len(ids); J = C.synthetic(n, int(rng.integers(1 << 30)))
    for k, t in enumerate(ids):
        tn = tens[t]; J.q[k] = tn["q"]; J.qhat[k] = tn["q"]
        J.rho[k] = max(rho_min, (a0 + tn["s"]) / (a0 + b0 + tn["s"] + tn["f"]))
    return J

def multiround(mode, seed, rounds=30, n_ten=150, zeta=1.0, prior=(4.0, 1.0), rho_min=0.05, kappa=0.0, mech="RAVA"):
    """mode: 'exploit' (10% build reputation for 10 rounds then default), 'whitewash' (unreliable tenants
    reset identity whenever rho < prior mean), 'normal'."""
    rng = np.random.default_rng(seed); tens = population(rng, n_ten); a0, b0 = prior
    if mode == "exploit":
        for t in rng.choice(n_ten, n_ten // 10, replace=False): tens[t].update(kind="exploit", q=0.97)
    W = []; lostX = []
    for r in range(rounds):
        if mode == "exploit" and r == 10:
            for tn in tens:
                if tn["kind"] == "exploit": tn["q"] = 0.10
        ids = [t for t in range(n_ten) if rng.random() < 0.6]
        J = epoch_jobs(rng, tens, ids, a0, b0, rho_min); b = J.bids(kappa)
        w = 1 / np.sqrt(J.size) if mech == "LOS" else C.rava_weight(J, kappa)
        st, nd, p = C.greedy(J, b, w, pay="none")
        draws = rng.random(J.n) < J.q
        W.append(float(J.v[(st >= 0) & draws].sum()))
        foot = J.L * J.K * (J.D / J.cap.sum(0)).sum(1) / (3 * J.T)
        lostX.append(float(sum(foot[k] for k, t in enumerate(ids) if st[k] >= 0 and not draws[k] and tens[t]["kind"] == "exploit")))
        for k, t in enumerate(ids):
            tn = tens[t]
            if st[k] >= 0:
                tn["s"] *= zeta; tn["f"] *= zeta
                if draws[k]: tn["s"] += 1
                else: tn["f"] += 1
            if mode == "whitewash" and tn["q"] < 0.7:
                rho = (a0 + tn["s"]) / (a0 + b0 + tn["s"] + tn["f"])
                if rho < a0 / (a0 + b0): tn["s"] = tn["f"] = 0.0        # new identity
    return W, lostX

def adversary_exp(seeds=20):
    res = {"exploit": {}, "whitewash": {}}
    for name, kw in [("LOS", dict(mech="LOS")), ("RAVA z=1", dict(zeta=1.0)), ("RAVA z=0.9", dict(zeta=0.9)), ("RAVA z=0.8", dict(zeta=0.8))]:
        res["exploit"][name] = [multiround("exploit", s, **kw) for s in range(seeds)]
    for name, kw in [("LOS", dict(mech="LOS")), ("RAVA no reset", dict(prior=(4.0, 1.0))),
                     ("RAVA prior 0.8", dict(prior=(4.0, 1.0), )), ("RAVA prior 0.5", dict(prior=(1.0, 1.0))), ("RAVA prior 0.3", dict(prior=(0.6, 1.4)))]:
        mode = "normal" if name in ("RAVA no reset",) else "whitewash"
        res["whitewash"][name] = [multiround(mode, s, **kw) for s in range(seeds)]
    json.dump(res, open("res_adversary.json", "w"))

def regret_exp(seeds=20, per=30, kappa=0.5):
    """Empirical single-round regret: utility of misreports minus truthful utility (should be <= 0)."""
    mult = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0]; gains = {f"bid x{m}": [] for m in mult}
    gains.update({"narrow window": [], "wider window": []})
    for s in range(seeds):
        J = C.synthetic(120, 120000 + s); b = J.bids(kappa); w = C.rava_weight(J, kappa)
        def util(i, bi, Jx, wx):
            bb = b.copy(); bb[i] = bi
            st, nd, p = C.greedy(Jx, bb, wx)
            if st[i] < 0: return 0.0
            ok = st[i] <= J.E[i] - J.L[i] + 1 and st[i] >= J.A[i]           # inside the TRUE window
            return (J.q[i] * (J.v[i] - p[i]) - (1 - J.q[i]) * kappa * p[i]) if ok else -kappa * p[i]
        rng = np.random.default_rng(s)
        for i in rng.choice(J.n, per, replace=False):
            u0 = util(i, b[i], J, w)
            for m in mult: gains[f"bid x{m}"].append(util(i, b[i] * m, J, w) - u0)
            if J.E[i] - J.A[i] + 1 > J.L[i]:
                J2 = C.Jobs(J.cap, J.T, J.D, J.K, J.L, J.A, J.E.copy(), J.v, J.q, J.rho, J.eta, J.u, J.cls)
                J2.E[i] -= 1; J2.phi = (J2.E - J2.A + 1 - J2.L) / (J2.E - J2.A + 1)
                gains["narrow window"].append(util(i, b[i], J2, C.rava_weight(J2, kappa)) - u0)
            J3 = C.Jobs(J.cap, J.T, J.D, J.K, J.L, J.A, J.E.copy(), J.v, J.q, J.rho, J.eta, J.u, J.cls)
            J3.E[i] = min(J.T - 1, J.E[i] + 4); J3.phi = (J3.E - J3.A + 1 - J3.L) / (J3.E - J3.A + 1)
            bb = b.copy(); st, nd, p = C.greedy(J3, bb, C.rava_weight(J3, kappa))
            # Assumption A3: outputs are released at the reported (later) deadline, so the job has no value
            uw = -(p[i] * (J.q[i] + (1 - J.q[i]) * kappa)) if st[i] >= 0 else 0.0
            gains["wider window"].append(uw - u0)
        print("regret", s, flush=True)
    json.dump(gains, open("res_regret.json", "w"))

def overbid_multiround(seeds=20, rounds=20, x=0.3, delta=0.9):
    """Forward-looking test: one tagged reliable tenant over-bids by (1+x) in round 0 to win and build
    reputation; discounted utility vs. truthful, common random numbers."""
    gains = []
    for s in range(seeds):
        for tag in range(5):
            tot = []
            for over in [0.0, x]:
                rng = np.random.default_rng(1000 * s + tag); tens = population(rng, 150)
                tens[tag].update(q=0.95); U = 0.0
                for r in range(rounds):
                    ids = [t for t in range(150) if rng.random() < 0.6 or t == tag]
                    J = epoch_jobs(rng, tens, ids, 4.0, 1.0, 0.05); b = J.bids(0.0); k0 = ids.index(tag)
                    if r == 0: b[k0] *= (1 + over)
                    st, nd, p = C.greedy(J, b, C.rava_weight(J, 0.0)); draws = rng.random(J.n) < J.q
                    if st[k0] >= 0 and draws[k0]: U += delta ** r * (J.v[k0] - p[k0])
                    for k, t in enumerate(ids):
                        if st[k] >= 0:
                            if draws[k]: tens[t]["s"] += 1
                            else: tens[t]["f"] += 1
                tot.append(U)
            gains.append(tot[1] - tot[0])
        print("overbid", s, flush=True)
    json.dump(gains, open("res_overbid.json", "w"))

def scale_exp():
    res = []
    for n, seeds in [(160, 5), (500, 5), (1000, 5), (2000, 3), (5000, 3), (10000, 2)]:
        for s in range(seeds):
            J = C.synthetic(n, 900000 + n + s, scale=n / 120); b = J.bids(0.0); row = dict(n=n, seed=s)
            for k, w in [("LOS", 1 / np.sqrt(J.size)), ("GVMPAC2", 1 / J.size), ("RAVA", C.rava_weight(J, 0.0))]:
                t = time.time(); st, nd, p = C.greedy(J, b, w, pay="none"); ta = time.time() - t
                t = time.time(); st, nd, p = C.greedy(J, b, w); tp = time.time() - t
                row[k] = dict(welfare=C.metrics(J, st, p, 0.0)["welfare"], t_alloc=ta, t_total=tp, served=int((st >= 0).sum()))
            dens = b / J.size; t = time.time(); st, nd, p = C.opd(J, b, np.percentile(dens, 5), np.percentile(dens, 95))
            row["OPD"] = dict(welfare=C.metrics(J, st, p, 0.0)["welfare"], t_total=time.time() - t)
            t = time.time(); row["bound"] = C.lp_bound(J, J.q * J.v); row["t_lp"] = time.time() - t
            res.append(row); print("scale", n, s, {k: round(v["t_total"], 2) for k, v in row.items() if isinstance(v, dict)}, flush=True)
            json.dump(res, open("res_scale.json", "w"))

def flex_exp(seeds=50):
    """Disclosure model: tenant i loses up to c_i of its value if started at the end of its true window
    (linear in delay). It discloses its true window iff lam*phi_i >= c_i/2, else it reports the tight
    window [a, a+L-1]. Welfare counts the delay loss."""
    res = []
    for lam in [0.0, 0.1, 0.2, 0.4, 0.8, "full", "none"]:
        for s in range(seeds):
            J = C.synthetic(120, 120000 + s); rng = np.random.default_rng(s + 99); c = rng.uniform(0, 0.3, J.n)
            slack = J.E - J.A + 1 - J.L; phi_true = J.phi.copy()
            if lam == "full": disc = np.ones(J.n, bool); lw = 0.0
            elif lam == "none": disc = np.zeros(J.n, bool); lw = 0.0
            else: disc = lam * phi_true >= c / 2; lw = lam
            J.E = np.where(disc, J.E, J.A + J.L - 1); J.phi = (J.E - J.A + 1 - J.L) / (J.E - J.A + 1)
            st, nd, p = C.greedy(J, J.bids(0.0), C.rava_weight(J, 0.0, lam=lw), pay="none")
            win = st >= 0; delay = np.where(win, st - J.A, 0)
            loss = np.where(slack > 0, c * delay / np.maximum(slack, 1), 0.0)
            res.append(dict(lam=str(lam), seed=s, welfare=float(np.sum((J.q * J.v * (1 - loss))[win])),
                            disclosed=float(disc.mean()), served=int(win.sum())))
        print("flex", lam, flush=True)
    json.dump(res, open("res_flex.json", "w"))

if __name__ == "__main__":
    {"main": main_exp, "kappa": kappa_exp, "bad": bad_exp, "adversary": adversary_exp, "regret": regret_exp,
     "overbid": overbid_multiround, "scale": scale_exp, "flex": flex_exp}[sys.argv[1]]()
