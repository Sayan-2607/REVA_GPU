import json, numpy as np, rava_sim as R
out = {}
# (c) expected GPU use over time for one congested instance, RAVA vs LOS
rng = np.random.default_rng(120003); jobs = R.gen_jobs(120, rng)
for j in jobs: j["b"] = j["v"]
def prof(order):
    a = R.place(order, jobs); used = np.zeros(R.T); idle = np.zeros(R.T); lost = np.zeros(R.T)
    for i, s in a.items():
        j = jobs[i]; g = j["d"][0]
        for t in range(s, s + j["L"]):
            used[t] += j["p"] * j["u"] * g; idle[t] += j["p"] * (1 - j["u"]) * g; lost[t] += (1 - j["p"]) * g
    return dict(used=used.tolist(), idle=idle.tolist(), lost=lost.tolist())
n = len(jobs)
out["profile"] = {"RAVA": prof(sorted(range(n), key=lambda k: -jobs[k]["b"] * R.rava_mult(jobs[k]))),
                  "LOS": prof(sorted(range(n), key=lambda k: -jobs[k]["b"] / np.sqrt(jobs[k]["D"]))),
                  "RAVA-E": prof(sorted(range(n), key=lambda k: -jobs[k]["b"] * R.rava_mult(jobs[k], gamma=1.0)))}
# (d) alpha x gamma grid
A = [0, 0.25, 0.5, 0.75, 1.0]; G = [0, 0.5, 1.0, 1.5, 2.0]
eff = np.zeros((5, 5)); util = np.zeros((5, 5))
for sd in range(10):
    rng = np.random.default_rng(7000 + sd); jobs = R.gen_jobs(120, rng)
    for j in jobs: j["b"] = j["v"]
    B = R.lp_bound(jobs, [j["p"] * j["v"] for j in jobs])
    for ia, a in enumerate(A):
        for ig, g in enumerate(G):
            o = sorted(range(120), key=lambda k: -jobs[k]["b"] * R.rava_mult(jobs[k], alpha=a, gamma=g))
            m = R.metrics(jobs, R.place(o, jobs), {})
            eff[ia, ig] += m["welfare"] / B * 10; util[ia, ig] += m["util"] * 10
out["grid"] = dict(A=A, G=G, eff=eff.tolist(), util=util.tolist())
json.dump(out, open("results_extra.json", "w"))
print(np.round(eff, 1)); print(np.round(util * 1, 1))
