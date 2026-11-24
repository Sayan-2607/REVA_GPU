"""Lemma 2 and Theorem 1: monotonicity, exact critical prices, individual rationality, no profitable lies."""
import numpy as np, pytest
from rava import core as C

SEEDS = range(6)


def inst(seed, n=40):
    return C.synthetic(n, 777000 + seed)


@pytest.mark.parametrize("seed", SEEDS)
def test_monotone_and_critical_price_exact(seed):
    J = inst(seed); b = J.bids(0.0); w = C.rava_weight(J, 0.0)
    st, _, p = C.greedy(J, b, w)
    rng = np.random.default_rng(seed)
    for i in rng.choice(np.where(st >= 0)[0], 5, replace=False):
        for mult in [1.1, 1.5, 3.0]:                     # higher bid keeps winning
            bb = b.copy(); bb[i] *= mult
            assert C.greedy(J, bb, w, pay="none")[0][i] >= 0
        lo, hi = 0.0, b[i]                               # brute-force threshold by bisection
        for _ in range(60):
            mid = (lo + hi) / 2; bb = b.copy(); bb[i] = mid
            if C.greedy(J, bb, w, pay="none")[0][i] >= 0: hi = mid
            else: lo = mid
        assert abs(hi - p[i]) <= 1e-6 * max(1, p[i])     # Algorithm 2 returns the exact threshold
        assert p[i] <= b[i] + 1e-9                        # individual rationality


@pytest.mark.parametrize("kappa", [0.0, 0.5])
@pytest.mark.parametrize("seed", SEEDS)
def test_no_profitable_bid_deviation(seed, kappa):
    J = inst(seed); b = J.bids(kappa); w = C.rava_weight(J, kappa)
    def util(i, bi):
        bb = b.copy(); bb[i] = bi; st, _, p = C.greedy(J, bb, w)
        return 0.0 if st[i] < 0 else J.q[i] * (J.v[i] - p[i]) - (1 - J.q[i]) * kappa * p[i]
    for i in np.random.default_rng(seed).choice(J.n, 8, replace=False):
        u0 = util(i, b[i])
        assert u0 >= -1e-9
        for m in [0.3, 0.7, 0.95, 1.05, 1.4, 3.0]:
            assert util(i, b[i] * m) <= u0 + 1e-9


def test_rava_reduces_to_los_and_second_price():
    """Proposition 3: kappa=1 (g=1), lambda=0, alpha=0.5 is LOS; alpha=0 is greedy second price."""
    J = inst(0); b = J.bids(1.0)
    a = C.greedy(J, b, C.rava_weight(J, 1.0, lam=0.0)); l = C.los(J, b)
    assert np.array_equal(a[0], l[0]) and np.allclose(a[2], l[2])
    a0 = C.greedy(J, b, C.rava_weight(J, 1.0, lam=0.0, alpha=0.0)); s = C.second_price(J, b)
    assert np.array_equal(a0[0], s[0])


def test_capacity_never_exceeded_multi_server():
    rng = np.random.default_rng(3); n = 60
    cap = np.array([[8.0]] * 4 + [[2.0]] * 4)
    g = rng.choice([1, 2, 4, 8, 16], n); K = np.where(g > 8, g // 8, 1); d = np.where(g > 8, 8, g)
    L = rng.integers(1, 6, n); A = rng.integers(0, 10, n); E = np.minimum(23, A + L - 1 + rng.integers(0, 6, n))
    J = C.Jobs(cap, 24, d[:, None].astype(float), K, L, A, E, rng.uniform(1, 10, n), np.ones(n), np.ones(n), np.ones(n), np.ones(n), np.zeros(n, int))
    st, nodes, _ = C.greedy(J, J.v, C.rava_weight(J, 0.0), pay="none")
    load = np.zeros((8, 24))
    for i in np.where(st >= 0)[0]:
        hs = nodes[i, :K[i]]; assert len(set(hs)) == K[i]
        for h in hs: load[h, st[i]:st[i] + L[i]] += d[i]
        assert A[i] <= st[i] and st[i] + L[i] - 1 <= E[i]
    assert np.all(load <= cap[:, :1] + 1e-9)
