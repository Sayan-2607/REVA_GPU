"""Lemma 1, Proposition 1, Lemma 3, Theorem 2, Propositions 4-5."""
import numpy as np
from rava import theory as th, effective_value, expected_utility, BetaReputation


def test_effective_value_is_indifference_price():
    rng = np.random.default_rng(0)
    for _ in range(200):
        v, q, k = rng.uniform(1, 100), rng.uniform(0.05, 1), rng.uniform(0, 1)
        t = effective_value(v, q, k)
        assert abs(expected_utility(v, q, k, t)) < 1e-9
        assert expected_utility(v, q, k, 0.9 * t) > 0 > expected_utility(v, q, k, 1.1 * t)


def test_rank_exactness_and_misspecification():
    rng = np.random.default_rng(1)
    for _ in range(200):
        q, k = rng.uniform(0.05, 1), rng.uniform(0, 1)
        assert np.isclose(effective_value(1.0, q, k) * th.g_kappa(q, k), q)          # calibrated: exact
        assert np.isclose(th.misspecification(q, q, k), 1.0)
        assert np.isclose(th.misspecification(q, rng.uniform(0.05, 1), 0.0), 1.0)     # kappa = 0: exact for any belief
        qh = min(1, q + rng.uniform(0.01, 0.5))                                          # overconfident: grows with kappa
        assert th.misspecification(q, qh, 0.2) <= th.misspecification(q, qh, 0.8) + 1e-12
        assert th.misspecification_naive(q, k) <= 1 + 1e-12


def test_epsilon_bound_shrinks_with_history():
    e = [th.epsilon_bound(0.9, 0.0, 100, 0.05, N) for N in [5, 50, 500]]
    assert e[0] > e[1] > e[2] and np.isclose(e[0] / e[2], 100)
    assert th.epsilon_bound(0.9, 1.0, 100, 0.05, 5) == 0.0                              # kappa = 1: exact


def test_exploit_bounds():
    assert th.defaults_before_threshold_no_forgetting(100, 0, 0.6) > th.defaults_before_threshold_no_forgetting(10, 0, 0.6)
    m = th.max_defaults_with_forgetting(0.6, zeta=0.8)
    rep = BetaReputation(zeta=0.8, rho_min=0.0)
    rep.counts["x"] = [1 / (1 - 0.8), 0.0]                                              # maximal honest history
    for _ in range(m): rep.update("x", False)
    assert rep.rho("x") < 0.6


def test_reputation_update_and_whitewash():
    r = BetaReputation(a0=4, b0=1, zeta=0.5, rho_min=0.1)
    assert np.isclose(r.rho("new"), 0.8)
    r.update("u", True); r.update("u", False)
    assert np.allclose(r.counts["u"], [0.5, 1.0])
    for _ in range(20): r.update("bad", False)
    assert r.rho("bad") >= 0.1 and r.whitewash_profitable("bad") and not r.whitewash_profitable("new")
