"""Baselines and bounds: LP bound dominates, VCG optimal for declared value, feasibility of every mechanism."""
import numpy as np, pytest
from rava import core as C


@pytest.mark.parametrize("seed", range(4))
def test_lp_bound_dominates_every_mechanism(seed):
    J = C.synthetic(60, 555000 + seed); b = J.bids(0.0); B = C.lp_bound(J, J.q * J.v)
    outs = [C.greedy(J, b, w, pay="none")[0] for w in [np.ones(J.n), 1 / np.sqrt(J.size), 1 / J.size, C.rava_weight(J, 0.0)]]
    outs.append(C.opd(J, b, np.percentile(b / J.size, 5), np.percentile(b / J.size, 95))[0]); outs.append(C.drf(J)[0])
    for st in outs:
        assert (J.q * J.v)[st >= 0].sum() <= B + 1e-6


def test_vcg_is_optimal_for_declared_value():
    J = C.synthetic(14, 424242); b = J.bids(0.0)
    st, p, ok = C.vcg(J, b); assert ok
    W = b[st >= 0].sum()
    for w in [np.ones(J.n), 1 / np.sqrt(J.size), C.rava_weight(J, 0.0)]:
        assert b[C.greedy(J, b, w, pay="none")[0] >= 0].sum() <= W + 1e-6
    assert np.all(p[st >= 0] <= b[st >= 0] + 1e-6)


def test_overbook_and_mcafee_run():
    J = C.synthetic(80, 99); b = J.bids(0.0)
    st, W, ev = C.overbook(J, b, 0.1, 10, 0); assert W >= 0 and ev >= 0
    st, _, p, unp = C.mcafee(J, b, np.random.default_rng(0)); assert unp >= 0
