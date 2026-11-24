"""Tune the OPD price bounds (percentiles of bid density) on validation data, then re-evaluate on test data."""
import json, numpy as np, pandas as pd
from rava import core as C
GRID = [(1, 99), (5, 95), (10, 90), (25, 75), (1, 50), (5, 50), (10, 50), (25, 50), (1, 25), (5, 25)]
def opd_w(J, lo, hi):
    b = J.bids(0.0); dens = b / J.size
    st, nd, p = C.opd(J, b, np.percentile(dens, lo), np.percentile(dens, hi)); return st, p
out = {}
# synthetic: validation seeds 500000+, test = the main-experiment seeds
for n in [40, 80, 120, 160]:
    val = {g: np.mean([C.metrics(J, *opd_w(J, *g), 0.0)["welfare"] for J in [C.synthetic(n, 500000 + n * 100 + s) for s in range(10)]]) for g in GRID}
    best = max(val, key=val.get)
    test = [C.metrics(J, *opd_w(J, *best), 0.0) for J in [C.synthetic(n, 1000 * n + s) for s in range(50)]]
    out[f"syn{n}"] = dict(best=best, test=test); print(n, best, flush=True)
# Philly: validation days < 2017-10-17
from rava.experiments import philly as P
P.DF = P.load()
def ph_eval(day, seed, g):
    J = P.epoch(P.DF, pd.Timestamp(day), seed); st, p = opd_w(J, *g); return P.realized(J, st, p, 0.0)["welfare"]
vd = [str(d.date()) for d in pd.date_range("2017-10-03", "2017-10-16")]
val = {g: np.mean([ph_eval(d, 1, g) for d in vd]) for g in GRID}; best = max(val, key=val.get)
R = json.load(open("res_philly_1-2.json")) + json.load(open("res_philly_3.json"))
out["philly"] = dict(best=best, test=[dict(day=r["day"], seed=r["seed"], welfare=ph_eval(r["day"], r["seed"], best)) for r in R])
json.dump(out, open("res_opd_tuned.json", "w"), default=list); print("philly", best)
