"""Several epochs: reputation learns who runs their jobs; settlement charges on use (Definition 1)."""
import numpy as np
from rava import RavaAuction, BetaReputation, charge, core as C

rng = np.random.default_rng(1)
true_q = {f"t{k}": (0.4 if k < 10 else 0.95) for k in range(50)}      # 10 unreliable tenants of 50
rep = BetaReputation(zeta=0.9)
auction = RavaAuction(kappa=0.0)
for epoch in range(8):
    users = [u for u in true_q if rng.random() < 0.7]
    J = C.synthetic(len(users), 1000 + epoch)
    for k, u in enumerate(users):
        J.q[k] = J.qhat[k] = true_q[u]; J.rho[k] = rep.rho(u)
    res = auction.run(J); value = revenue = 0.0
    for k in res.winners:
        ran = rng.random() < J.q[k]
        value += J.v[k] * ran; revenue += charge(res.prices[k], ran, auction.kappa); rep.update(users[k], ran)
    bad = np.mean([rep.rho(u) for u in true_q if true_q[u] < 0.5]); good = np.mean([rep.rho(u) for u in true_q if true_q[u] > 0.5])
    print(f"epoch {epoch}: realised value {value:7.1f}  revenue {revenue:6.1f}  mean rho unreliable {bad:.2f} / reliable {good:.2f}")
