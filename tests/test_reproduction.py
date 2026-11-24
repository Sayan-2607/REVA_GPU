"""Regression against the stored raw results of the paper."""
import json, os, numpy as np, pytest
from rava import core as C
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results")


@pytest.mark.parametrize("n,s", [(40, 0), (120, 7), (160, 49)])
def test_synthetic_main_regression(n, s):
    stored = json.load(open(os.path.join(RES, "res_main.json")))[str(n)][s]
    J = C.synthetic(n, 1000 * n + s); b = J.bids(0.0)
    for name, w in [("RAVA", C.rava_weight(J, 0.0)), ("LOS", 1 / np.sqrt(J.size)), ("GVMPAC2", 1 / J.size)]:
        st, _, p = C.greedy(J, b, w); m = C.metrics(J, st, p, 0.0)
        assert np.isclose(m["welfare"], stored[name]["welfare"]) and np.isclose(m["revenue"], stored[name]["revenue"])


def test_paper_claims():
    from rava.cli import cmd_check
    class A: results = RES
    assert cmd_check(A) == 0


@pytest.mark.slow
def test_philly_epoch_regression():
    import pandas as pd
    os.environ["RAVA_DATA"] = os.path.join(ROOT, "data")
    from rava.experiments import philly as P
    P.DF = P.load()
    stored = {(r["day"], r["seed"]): r for r in json.load(open(os.path.join(RES, "res_philly_1-2.json")))}
    o = P.run(pd.Timestamp("2017-11-15"), 2); s = stored[("2017-11-15", 2)]
    for k in ["RAVA", "LOS", "SPA", "GVMPAC2"]:
        assert np.isclose(o[k]["welfare"], s[k]["welfare"])
