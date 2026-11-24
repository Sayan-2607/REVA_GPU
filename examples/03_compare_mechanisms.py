"""Compare RAVA with the paper's baselines on one synthetic workload (Section VIII-B setting)."""
import numpy as np
from rava import core as C

J = C.synthetic(120, 120000); b = J.bids(0.0); bound = C.lp_bound(J, J.q * J.v)
rng = np.random.default_rng(0)
runs = {
    "First-price": C.first_price(J, b, rng.uniform(0.6, 0.9, J.n))[0],
    "Second-price": C.second_price(J, b)[0], "LOS": C.los(J, b)[0], "G-VMPAC-II": C.gvmpac2(J, b)[0],
    "Online primal-dual": C.opd(J, b, np.percentile(b / J.size, 5), np.percentile(b / J.size, 95))[0],
    "DRF": C.drf(J)[0], "RAVA": C.greedy(J, b, C.rava_weight(J, 0.0))[0],
}
for name, st in runs.items():
    print(f"{name:20s} efficiency {100 * (J.q * J.v)[st >= 0].sum() / bound:5.1f}%   jobs served {int((st >= 0).sum())}")
st, W, ev = C.overbook(J, b, 0.1, 30, 0); print(f"{'Overbooking':20s} efficiency {100 * W / bound:5.1f}%")
