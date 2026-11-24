"""Command-line interface:  rava <command> [options]

  rava example [--j1-bid 130]                  worked example of Section VI (all mechanisms)
  rava experiment {main,kappa,bad,flex,regret,adversary,overbid,scale} [--results DIR]
  rava philly --seeds 1,2 [--killed-ok]       trace-driven GPU experiment (Section VIII-C)
  rava philly-overbook | philly-combo | tune-opd
  rava analyze [--results DIR] [--figures DIR]  every number (summary.json) and figure of the paper
  rava check [--results DIR]                   compare summary.json with the numbers stated in the paper
  rava theory {eps,exploit,mis} ...            closed-form results of Section V
  rava slurm {auction,settle} ...              Slurm broker (Section VII)
  rava prepare-philly LOG OUT | philly-util TRACE_DIR
"""
import argparse, os, sys, runpy, json, contextlib
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


@contextlib.contextmanager
def _in(dirpath, argv, env=None):
    old_cwd, old_argv, old_env = os.getcwd(), sys.argv[:], dict(os.environ)
    os.makedirs(dirpath, exist_ok=True); os.chdir(dirpath); sys.argv = argv
    if env: os.environ.update(env)
    try:
        yield
    finally:
        os.chdir(old_cwd); sys.argv = old_argv; os.environ.clear(); os.environ.update(old_env)


def _run_module(mod, argv, results, data=None):
    env = {"RAVA_DATA": os.path.abspath(data)} if data else None
    with _in(os.path.abspath(results), [mod] + argv, env):
        runpy.run_module(mod, run_name="__main__")


def cmd_example(a):
    from rava.workloads.example import worked_example
    from rava import core as C
    J = worked_example(a.j1_bid); b = J.v.copy(); w = C.rava_weight(J, 0.0)
    print(f"{'Job':4s} {'bid':>6s} {'rho':>5s} {'D':>6s} {'phi':>5s} {'w':>6s} {'score':>7s}")
    for i in range(J.n):
        print(f"{J.names[i]:4s} {b[i]:6.0f} {J.rho[i]:5.2f} {J.size[i]:6.3f} {J.phi[i]:5.2f} {w[i]:6.3f} {b[i] * w[i]:7.2f}")
    rows = []
    st, _, p = C.greedy(J, b, np.ones(J.n), pay="bid"); rows.append(("First-price (bids as given)", st, p))
    st, _, p = C.second_price(J, b); rows.append(("Second-price", st, p))
    st, _, p = C.los(J, b); rows.append(("LOS", st, p))
    st, p, ok = C.vcg(J, b); rows.append(("VCG (exact)", st, p))
    st, _, p = C.greedy(J, b, w); rows.append(("RAVA (r=0)", st, p))
    st, _, p3 = C.greedy(J, b, w, reserve=3.0); rows.append(("RAVA (r=3)", st, p3))
    print(f"\n{'Mechanism':28s} {'winners':22s} {'declared':>8s} {'E[SW]':>7s} {'sum p':>7s}")
    for name, st, p in rows:
        win = np.where(st >= 0)[0]
        print(f"{name:28s} {','.join(J.names[i] for i in win):22s} {J.v[win].sum():8.0f} {(J.rho[win] * J.v[win]).sum():7.1f} {p[win].sum():7.2f}")
    st, _, p = C.greedy(J, b, w)
    print("\nRAVA prices:", {J.names[i]: round(float(p[i]), 2) for i in np.where(st >= 0)[0]},
          "\nRAVA starts:", {J.names[i]: int(st[i]) + 1 for i in np.where(st >= 0)[0]}, "(slot numbers 1-4)")


def cmd_theory(a):
    from rava import theory as th
    if a.what == "eps":
        e = th.epsilon_bound(a.delta, a.kappa, a.vbar, a.rho_min, a.mass, a.a0, a.b0, a.zeta)
        print(f"Theorem 2 epsilon bound = {e:.6g}  (Lambda = {th.lipschitz(a.kappa, a.vbar, a.rho_min):.6g})")
    elif a.what == "exploit":
        print("zeta=1  defaults while rho >= rho_bar:", th.defaults_before_threshold_no_forgetting(a.s, a.f, a.rho_bar, a.a0, a.b0))
        print(f"zeta={a.zeta} history-independent bound m*:", th.max_defaults_with_forgetting(a.rho_bar, a.a0, a.b0, a.zeta))
    elif a.what == "mis":
        print(f"Proposition 1: mu (RAVA) = {th.misspecification(a.q, a.qhat, a.kappa):.4f}, mu (naive) = {th.misspecification_naive(a.qhat, a.kappa):.4f}")


CLAIMS = [  # (path in summary.json, value stated in the paper, tolerance)
    (("main", "120", "LOS", "gain"), 2.94, 0.01), (("main", "120", "RAVA", "eff"), 85.0, 0.05),
    (("main", "40", "LOS", "gain"), 2.15, 0.01), (("main", "160", "LOS", "gain"), 2.96, 0.01),
    (("vcg40", "eff"), 91.6, 0.05), (("scatter", "wins"), 136, 0),
    (("philly", "tab", "LOS", "gain"), 6.35, 0.01), (("philly", "tab", "RAVA", "eff"), 86.1, 0.05),
    (("philly", "tab", "Overbook", "gain"), 1.06, 0.01), (("philly", "tab", "RAVA+OB0.2", "eff"), 91.1, 0.05),
    (("philly", "combo_vs_losob", "gain"), 5.01, 0.01), (("philly", "killed_ok", "gain"), -0.02, 0.01),
    (("exploit", "rava_vs_los_lost", "gain"), -18.0, 0.05), (("overbid", "mean"), -0.42, 0.01),
    (("flex", "0.2", "disclosed"), 54.4, 0.05), (("scale", "gain10k", "gain"), 2.53, 0.01),
]


def cmd_check(a):
    S = json.load(open(os.path.join(a.results, "summary.json"))); bad = 0
    for path, val, tol in CLAIMS:
        x = S
        for k in path: x = x[k]
        ok = abs(round(float(x), 2) - val) <= tol + 1e-9
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {'/'.join(path):40s} paper={val:>8}  computed={float(x):.3f}")
    rg = S["regret"]; pos = sum(v["pos"] for v in rg.values()); n = sum(v["n"] for v in rg.values())
    print(f"{'OK  ' if pos == 0 else 'FAIL'} regret: {pos} profitable deviations out of {n} (paper: 0 of 5,291)"); bad += pos != 0
    print("all claims reproduced" if bad == 0 else f"{bad} mismatches"); return bad


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rava", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("example"); e.add_argument("--j1-bid", type=float, default=130)
    x = sp.add_parser("experiment"); x.add_argument("name", choices=["main", "kappa", "bad", "flex", "regret", "adversary", "overbid", "scale"])
    x.add_argument("--results", default=os.path.join(REPO, "results"))
    ph = sp.add_parser("philly"); ph.add_argument("--seeds", default="1"); ph.add_argument("--killed-ok", action="store_true")
    for name in ["philly", "philly-overbook", "philly-combo", "tune-opd"]:
        p_ = ph if name == "philly" else sp.add_parser(name)
        p_.add_argument("--results", default=os.path.join(REPO, "results")); p_.add_argument("--data", default=os.path.join(REPO, "data"))
    an = sp.add_parser("analyze"); an.add_argument("--results", default=os.path.join(REPO, "results")); an.add_argument("--figures", default=os.path.join(REPO, "paper", "figures"))
    ck = sp.add_parser("check"); ck.add_argument("--results", default=os.path.join(REPO, "results"))
    th = sp.add_parser("theory"); th.add_argument("what", choices=["eps", "exploit", "mis"])
    for k, v in dict(delta=0.9, kappa=0.0, vbar=100.0, rho_min=0.05, mass=10.0, a0=4.0, b0=1.0, zeta=1.0, s=20.0, f=0.0, rho_bar=0.6, q=0.5, qhat=0.8).items():
        th.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    sl = sp.add_parser("slurm", add_help=False); sl.add_argument("rest", nargs=argparse.REMAINDER)
    pp = sp.add_parser("prepare-philly"); pp.add_argument("log"); pp.add_argument("out")
    pu = sp.add_parser("philly-util"); pu.add_argument("trace_dir")
    a = ap.parse_args(argv)
    if a.cmd == "example": cmd_example(a)
    elif a.cmd == "experiment": _run_module("rava.experiments.synthetic", [a.name], a.results)
    elif a.cmd == "philly": _run_module("rava.experiments.philly", [a.seeds] + (["killed_ok"] if a.killed_ok else []), a.results, a.data)
    elif a.cmd == "philly-overbook": _run_module("rava.experiments.philly_overbook", [], a.results, a.data)
    elif a.cmd == "philly-combo": _run_module("rava.experiments.combo", [], a.results, a.data)
    elif a.cmd == "tune-opd": _run_module("rava.experiments.tune_opd", [], a.results, a.data)
    elif a.cmd == "analyze":
        with _in(os.path.abspath(a.results), ["figures", os.path.abspath(a.results), os.path.abspath(a.figures)]):
            runpy.run_module("rava.analysis.figures", run_name="__main__")
    elif a.cmd == "check": sys.exit(cmd_check(a))
    elif a.cmd == "theory": cmd_theory(a)
    elif a.cmd == "slurm":
        from rava.integrations import slurm_broker; slurm_broker.main(a.rest)
    elif a.cmd == "prepare-philly":
        sys.argv = ["prepare", a.log, a.out]; runpy.run_module("rava.experiments.prepare_philly", run_name="__main__")
    elif a.cmd == "philly-util":
        sys.argv = ["util", a.trace_dir]; runpy.run_module("rava.experiments.philly_util", run_name="__main__")


if __name__ == "__main__":
    main()
