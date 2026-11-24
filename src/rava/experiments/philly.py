"""
Trace-driven GPU experiment on the Microsoft Philly trace (Jeon et al., USENIX ATC 2019).

One epoch = one calendar day of submissions (2017-10-03 .. 2017-11-30). The auction sells a 160-GPU
sub-cluster (16 x 8-GPU + 16 x 2-GPU servers) over a 48-hour horizon of hourly slots.
Real from the trace : GPUs per job, gang size, run time, submission hour, user id, outcome (Pass /
                      Failed / Killed), each user's earlier history, measured GPU utilisation.
Synthetic (stated)  : values v (GPU-hours x user and job log-normal multipliers), deadline slack,
                      subsampling rate, reservation length of failed jobs (user's median passed length).
"""
import sys, json, numpy as np, pandas as pd
from rava import core as C

T, NODES8, NODES2, FRAC = 48, 16, 16, 0.35

def load():
    import os
    D = os.environ.get("RAVA_DATA", "../data")
    df = pd.read_csv(f"{D}/philly_jobs.csv.gz", parse_dates=["submit"])
    ut = pd.read_csv(f"{D}/philly_job_util.csv.gz")[["jobid", "gpu_util"]]
    df = df.merge(ut, on="jobid", how="left")
    df["ok"] = (df.status == "Pass").astype(int)
    df = df.sort_values("submit").reset_index(drop=True)
    return df

def user_stats(hist, a0=3.0, b0=1.0):
    g = hist.groupby("user")
    s = g.ok.sum(); n = g.ok.count(); util = g.gpu_util.mean() / 100.0
    rho = (a0 + s) / (a0 + b0 + n)
    return rho, util

def epoch(df, day, seed, killed_ok=False, kappa=0.0):
    rng = np.random.default_rng(seed)
    hist = df[df.submit < day]
    if killed_ok:                                                   # history uses the same success definition
        hist = hist.assign(ok=hist.status.isin(["Pass", "Killed"]).astype(int))
    rho_u, util_u = user_stats(hist)
    glob_util = float(np.nanmean(hist.gpu_util)) / 100.0
    today = df[(df.submit >= day) & (df.submit < day + pd.Timedelta(days=1))]
    today = today.sample(frac=FRAC, random_state=seed).sort_values("submit")
    passed = hist[hist.ok == 1]
    med_user = passed.groupby(["user", "gpus"]).hours.median(); med_g = passed.groupby("gpus").hours.median()
    rows = []
    for r in today.itertuples():
        g = int(r.gpus)
        if g < 1 or g > 64: continue
        h = r.hours
        if r.status != "Pass":                                     # reservation length for failed jobs
            h = med_user.get((r.user, g), med_g.get(g, h))
        L = int(np.clip(np.ceil(max(h, 1e-3)), 1, 24))
        K, d = (1, g) if g <= 8 else (int(np.ceil(g / 8)), 8)
        a = int(r.submit.hour); slack = int(rng.integers(0, 13)); e = min(T - 1, a + L - 1 + slack)
        if a + L > T: continue
        ok = 1 if r.status == "Pass" or (killed_ok and r.status == "Killed") else 0
        rows.append((g, K, d, L, a, e, r.user, ok, rho_u.get(r.user, 0.75),
                     util_u.get(r.user, glob_util) if not np.isnan(util_u.get(r.user, np.nan)) else glob_util,
                     (r.gpu_util / 100.0) if not np.isnan(r.gpu_util) else util_u.get(r.user, glob_util)))
    R = pd.DataFrame(rows, columns=["g", "K", "d", "L", "a", "e", "user", "ok", "rho", "eta", "u"])
    R["u"] = R.u.fillna(glob_util).clip(0.01, 1); R["eta"] = R.eta.fillna(glob_util).clip(0.01, 1)
    users = R.user.unique(); umul = dict(zip(users, rng.lognormal(0, 0.5, len(users))))
    v = R.g * R.L * R.user.map(umul).values * rng.lognormal(0, 0.25, len(R)) * 10
    cap = np.array([[8.0]] * NODES8 + [[2.0]] * NODES2)
    cls = np.where(R.g == 1, 0, np.where(R.g <= 8, 1, 2))
    uid = pd.factorize(R.user)[0]
    J = C.Jobs(cap, T, R.d.values[:, None].astype(float), R.K.values, R.L.values, R.a.values, R.e.values,
               v.values, R.ok.values.astype(float), R.rho.values, R.eta.values, R.u.values, cls, user=uid)
    return J

def realized(J, start, p, kappa):
    win = start >= 0; ok = J.q > 0.5
    gh = J.L * J.K * J.D[:, 0]; capgh = J.cap[:, 0].sum() * J.T
    return dict(welfare=float(J.v[win & ok].sum()), revenue=float(p[win & ok].sum() + kappa * p[win & ~ok].sum()),
                served=int(win.sum()), useful_gpu=float((J.u * gh)[win & ok].sum() / capgh),
                lost_gpu=float(gh[win & ~ok].sum() / capgh), reserved_gpu=float(gh[win].sum() / capgh),
                jain=C.metrics(J, start, p, kappa)["jain"])

def run(day, seed, killed_ok=False):
    df = DF
    J = epoch(df, day, seed, killed_ok)
    b = J.v.copy()                                                # kappa = 0: dominant bid is v
    out = {"day": str(day.date()), "seed": seed, "n": J.n}
    los = 1 / np.sqrt(J.size)
    mech = {
        "SPA": (b, np.ones(J.n)), "LOS": (b, los), "GVMPAC2": (b, 1 / J.size),
        "RAVA": (b, C.rava_weight(J, 0.0)), "RAVA-E": (b, C.rava_weight(J, 0.0, gamma=1.0)),
        "RAVA-no-rho": (b, C.rava_weight(J, 0.0, use_rho=False)),
        "RAVA-a1": (b, C.rava_weight(J, 0.0, alpha=1.0)),
    }
    for k, (bb, w) in mech.items():
        st, nd, p = C.greedy(J, bb, w); out[k] = realized(J, st, p, 0.0)
    rng = np.random.default_rng(seed + 7)
    shade = rng.uniform(0.6, 0.9, J.n)
    st, nd, p = C.greedy(J, b * shade, np.ones(J.n), pay="bid"); out["FPA"] = realized(J, st, p, 0.0)
    dens = b / J.size; st, nd, p = C.opd(J, b, np.percentile(dens, 5), np.percentile(dens, 95)); out["OPD"] = realized(J, st, p, 0.0)
    st, _, _ = C.drf(J); out["DRF"] = realized(J, st, np.zeros(J.n), 0.0)
    st, _, p, unp = C.mcafee(J, b, rng); out["McAfee"] = realized(J, st, p, 0.0); out["McAfee"]["unplaced"] = unp
    # overbooking with the real outcome: failed jobs release their slots, shown jobs may be evicted
    w = los; order = np.argsort(-b * w, kind="stable").astype(np.int64)
    st, nodes = C.place(order, J.cap * 1.1, J.T, J.D, J.K, J.L, J.A, J.E)
    shown = [i for i in order if st[i] >= 0 and J.q[i] > 0.5]
    load = np.zeros((J.cap.shape[0], J.T, 1)); kept = set(shown)
    for i in shown:
        for nd_ in nodes[i, :J.K[i]]: load[nd_, st[i]:st[i] + J.L[i]] += J.D[i]
    for i in reversed(shown):
        if any(np.any(load[nd_, st[i]:st[i] + J.L[i]] > J.cap[nd_] + 1e-9) for nd_ in nodes[i, :J.K[i]]):
            kept.discard(i)
            for nd_ in nodes[i, :J.K[i]]: load[nd_, st[i]:st[i] + J.L[i]] -= J.D[i]
    st2 = st.copy(); okv = J.q.copy()
    for i in shown:
        if i not in kept: okv[i] = 0.0                            # evicted: no value
    J2 = C.Jobs(J.cap, J.T, J.D, J.K, J.L, J.A, J.E, J.v, okv, J.rho, J.eta, J.u, J.cls, user=J.user)
    out["Overbook"] = realized(J2, st2, np.zeros(J.n), 0.0); out["Overbook"]["evicted"] = len(shown) - len(kept)
    out["grid"] = {}
    for a in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for g in [0.0, 0.5, 1.0]:
            st, nd, p = C.greedy(J, b, C.rava_weight(J, 0.0, alpha=a, gamma=g), pay="none")
            out["grid"][f"{a}_{g}"] = realized(J, st, p, 0.0)
    out["bound"] = C.lp_bound(J, J.v * J.q)                       # oracle LP bound (knows outcomes)
    return out

if __name__ == "__main__":
    DF = load()
    days = pd.date_range("2017-10-03", "2017-11-30", freq="D")
    seeds = [int(x) for x in sys.argv[1].split(",")]
    killed_ok = len(sys.argv) > 2 and sys.argv[2] == "killed_ok"
    res = []
    for d in days:
        for s in seeds:
            res.append(run(d, s, killed_ok)); print(d.date(), s, res[-1]["n"], flush=True)
    json.dump(res, open(f"res_philly{'_killedok' if killed_ok else ''}_{sys.argv[1].replace(',', '-')}.json", "w"))
