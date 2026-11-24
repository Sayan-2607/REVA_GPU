"""Quick start: describe a cluster and some jobs, run RAVA, read winners and prices."""
from rava import RavaAuction, BetaReputation, jobs_from_records

# One server with 8 GPUs, 64 cores, 256 GB; one epoch of 6 hourly slots.
cluster = [[8, 64, 256]]
rep = BetaReputation()                       # Beta(4,1) prior; tenants with history get their own rho
rep.counts.update({"alice": [18, 1], "bob": [2, 6], "carol": [9, 0]})

jobs = [
    dict(user="alice", demand=[4, 16, 64], slots=3, window=(0, 5), value=60),
    dict(user="bob",   demand=[8, 32, 128], slots=4, window=(0, 5), value=150),   # big, unreliable
    dict(user="carol", demand=[2, 8, 32],  slots=2, window=(1, 3), value=25),
    dict(user="dave",  demand=[2, 8, 32],  slots=6, window=(0, 5), value=30),     # newcomer
]
J = jobs_from_records(cluster, jobs, T=6, reputation=rep)
auction = RavaAuction(kappa=0.0, alpha=0.5, gamma=0.0, lam=0.2)
res = auction.run(J)
for i, r in enumerate(jobs):
    status = f"wins slot {res.start[i]}, pays {res.prices[i]:.2f}" if res.start[i] >= 0 else "loses"
    print(f"{r['user']:6s} bid={res.bids[i]:6.1f} rho={J.rho[i]:.2f} score={res.scores[i]:7.2f}  {status}")
print("expected welfare:", round(auction.evaluate(J, res)["welfare"], 2))
