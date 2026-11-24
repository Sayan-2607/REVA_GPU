"""Overbooking baseline with phantom servers for integer GPU capacities (Philly)."""
import sys, json, numpy as np, pandas as pd
from rava import core as C
from rava.experiments import philly as P

def overbook_phantom(J, b, tau=0.1, seed=0):
    rng = np.random.default_rng(seed); N = J.cap.shape[0]
    extra = rng.choice(N, int(np.ceil(tau * N)), replace=False)
    cap2 = np.vstack([J.cap, J.cap[extra]])
    order = np.argsort(-b / np.sqrt(J.size), kind="stable").astype(np.int64)
    st, nodes = C.place(order, cap2, J.T, J.D, J.K, J.L, J.A, J.E)
    R = C.init_R(J.cap, J.T); out = np.empty(max(J.K.max(), 1), np.int64); ok = np.zeros(J.n, bool); ev = 0
    for i in order:                                    # jobs that actually run, in rank order, at their reserved start
        if st[i] < 0 or J.q[i] < 0.5: continue
        s = C.try_place(R, J.D[i], J.K[i], J.L[i], st[i], st[i] + J.L[i] - 1, out, True)
        if s >= 0: ok[i] = True
        else: ev += 1
    return st, ok, ev

if __name__ == "__main__":
    P.DF = P.load(); res = []
    for f, killed in [("res_philly_1-2.json", False), ("res_philly_3.json", False), ("res_philly_killedok_1.json", True)]:
        for r in json.load(open(f)):
            J = P.epoch(P.DF, pd.Timestamp(r["day"]), r["seed"], killed); assert J.n == r["n"]
            st, ok, ev = overbook_phantom(J, J.v.copy(), 0.1, r["seed"])
            J2 = C.Jobs(J.cap, J.T, J.D, J.K, J.L, J.A, J.E, J.v, ok.astype(float), J.rho, J.eta, J.u, J.cls, user=J.user)
            m = P.realized(J2, st, np.zeros(J.n), 0.0); m["evicted"] = ev
            m["lost_gpu"] = P.realized(J, st, np.zeros(J.n), 0.0)["lost_gpu"]
            res.append(dict(file=f, day=r["day"], seed=r["seed"], Overbook=m))
    json.dump(res, open("res_philly_overbook.json", "w")); print("done", len(res))
