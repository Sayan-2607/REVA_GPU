"""RAVA ranking combined with overbooking (phantom servers on Philly)."""
import json, numpy as np, pandas as pd
from rava import core as C
from rava.experiments import philly as P, philly_overbook as PO
def ob_rank(J, score, tau, seed):
    rng = np.random.default_rng(seed); N = J.cap.shape[0]; extra = rng.choice(N, int(np.ceil(tau * N)), replace=False)
    order = np.argsort(-score, kind="stable").astype(np.int64)
    st, nodes = C.place(order, np.vstack([J.cap, J.cap[extra]]), J.T, J.D, J.K, J.L, J.A, J.E)
    R = C.init_R(J.cap, J.T); out = np.empty(max(J.K.max(), 1), np.int64); ok = np.zeros(J.n, bool)
    for i in order:
        if st[i] >= 0 and J.q[i] > 0.5:
            ok[i] = C.try_place(R, J.D[i], J.K[i], J.L[i], st[i], st[i] + J.L[i] - 1, out, True) >= 0
    J2 = C.Jobs(J.cap, J.T, J.D, J.K, J.L, J.A, J.E, J.v, ok.astype(float), J.rho, J.eta, J.u, J.cls, user=J.user)
    m = P.realized(J2, st, np.zeros(J.n), 0.0); m["lost_gpu"] = P.realized(J, st, np.zeros(J.n), 0.0)["lost_gpu"]; return m
P.DF = P.load(); res = []
R = json.load(open("res_philly_1-2.json")) + json.load(open("res_philly_3.json"))
for r in R:
    J = P.epoch(P.DF, pd.Timestamp(r["day"]), r["seed"]); b = J.v
    res.append(dict(day=r["day"], seed=r["seed"], **{f"RAVA+OB{t}": ob_rank(J, b * C.rava_weight(J, 0.0), t, r["seed"]) for t in [0.1, 0.2]},
                    **{f"LOS+OB{t}": ob_rank(J, b / np.sqrt(J.size), t, r["seed"]) for t in [0.2]}))
json.dump(res, open("res_philly_combo.json", "w")); print("done")
