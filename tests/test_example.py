"""Section VI worked example: every number in Tables III and the text."""
import numpy as np, pytest
from rava import core as C
from rava.workloads.example import worked_example


@pytest.fixture(scope="module")
def J():
    return worked_example(130)


def test_scores(J):
    s = J.v * C.rava_weight(J, 0.0)
    assert np.allclose(np.round(s, 2), [34.00, 35.47, 29.40, 28.00, 13.58, 14.31])


def test_rava_allocation_and_prices(J):
    st, _, p = C.greedy(J, J.v, C.rava_weight(J, 0.0))
    assert [J.names[i] for i in np.where(st >= 0)[0]] == ["J2", "J3", "J4", "J5", "J6"]
    assert round(p[1], 2) == 46.02 and round(p[2], 2) == 28.57
    assert np.isclose((J.rho * J.v)[st >= 0].sum(), 148.7, atol=0.05)


def test_reserve_price(J):
    st, _, p = C.greedy(J, J.v, C.rava_weight(J, 0.0), reserve=3.0)
    assert np.isclose(p[st >= 0].sum(), 92.59, atol=0.01)


def test_price_based_mechanisms_pick_J1(J):
    for st, _, p in [C.second_price(J, J.v), C.los(J, J.v)]:
        assert [J.names[i] for i in np.where(st >= 0)[0]] == ["J1", "J3"]
    st, p, ok = C.vcg(J, J.v)
    assert ok and [J.names[i] for i in np.where(st >= 0)[0]] == ["J1", "J3"]
    assert np.allclose(p[[0, 2]], [128, 22])


def test_low_j1_bid_vcg_matches_rava():
    J = worked_example(90)
    st, p, ok = C.vcg(J, J.v)
    assert [J.names[i] for i in np.where(st >= 0)[0]] == ["J2", "J3", "J4", "J5", "J6"]
    assert np.isclose(p.sum(), 12.0)
    st2, _, p2 = C.greedy(J, J.v, C.rava_weight(J, 0.0), reserve=3.0)
    assert np.isclose(p2[st2 >= 0].sum(), 52.57, atol=0.01)
