"""Compute every number and figure in the paper from the raw result files.
Writes summary.json (numbers quoted in the text and tables) and figures/*.pdf."""
import json, os, numpy as np, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from rava import stats as S

plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amssymb}",
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "grid.linewidth": 0.4, "grid.alpha": 0.5,
    "lines.linewidth": 1.2, "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL, DCOL = 3.5, 7.16
BLUE, GREY, DGREY, LGREY, RED, GREEN, AMBER = "#1f5fbf", "#9a9a9a", "#555555", "#c8c8c8", "#c0392b", "#2e8b57", "#e0a030"
import sys
RES = sys.argv[1] if len(sys.argv) > 1 else "../results"
FIG = (os.path.abspath(sys.argv[2]) + "/") if len(sys.argv) > 2 else os.path.abspath(os.path.join(RES, "../paper/figures")) + "/"
os.chdir(RES)
os.makedirs(FIG, exist_ok=True)
OUT = {}
NAME = {"FPA": "First-price", "SPA": "Second-price", "LOS": "LOS", "GVMPAC2": "G-VMPAC-II", "OPD": "Online primal-dual",
        "McAfee": "McAfee DA", "DRF": "DRF", "Overbook": "Overbooking", "RAVA": "RAVA", "RAVA-E": r"RAVA-E ($\gamma{=}1$)",
        "RAVA-a1": r"RAVA ($\alpha{=}1$)", "RAVA-no-rho": r"RAVA w/o $g(\rho)$", "RAVA-no-phi": r"RAVA w/o flex.", "RAVA-no-size": r"RAVA ($\alpha{=}0$)"}

# ============================================================ synthetic main (50 seeds)
M = json.load(open("res_main.json")); VCG = json.load(open("res_vcg_v1.json"))
base = ["FPA", "SPA", "LOS", "GVMPAC2", "OPD", "McAfee", "DRF", "Overbook"]
OUT["main"] = {}
for n, runs in M.items():
    tab = {}; ps = []; keys = []
    for m in base + ["RAVA-E", "RAVA-a1", "RAVA-no-rho", "RAVA-no-phi", "RAVA-no-size"]:
        st = S.paired([r["RAVA"]["welfare"] for r in runs], [r[m]["welfare"] for r in runs]); ps.append(st["p_wilcoxon"]); keys.append(m); tab[m] = st
    adj = S.holm(ps)
    for k, a in zip(keys, adj): tab[k]["p_holm"] = float(a)
    for m in base + ["RAVA", "RAVA-E", "RAVA-a1", "RAVA-no-rho", "RAVA-no-phi", "RAVA-no-size"]:
        e = [r[m]["welfare"] / r["bound"] * 100 for r in runs]
        tab.setdefault(m, {}).update(eff=float(np.mean(e)), eff_ci=float(1.96 * np.std(e, ddof=1) / np.sqrt(len(e))),
            revenue=float(np.mean([r[m]["revenue"] for r in runs])), util=float(np.mean([r[m]["util"] for r in runs]) * 100),
            waste=float(np.mean([r[m]["reserved"] - r[m]["util"] for r in runs]) * 100), jain=float(np.mean([r[m]["jain"] for r in runs])),
            served=float(np.mean([r[m]["served"] for r in runs])), t=float(np.mean([r[m].get("t", np.nan) for r in runs])))
    OUT["main"][n] = tab
ve = [VCG[s]["VCG"]["welfare"] / M["40"][s]["bound"] * 100 for s in range(len(VCG))]
OUT["vcg40"] = dict(eff=float(np.mean(ve)), ci=float(1.96 * np.std(ve, ddof=1) / np.sqrt(len(ve))), t=float(np.mean([v["VCG"]["t"] for v in VCG])),
                    tmax=float(np.max([v["VCG"]["t"] for v in VCG])),
                    rava_vs=S.paired([M["40"][s]["RAVA"]["welfare"] for s in range(len(VCG))], [VCG[s]["VCG"]["welfare"] for s in range(len(VCG))]))

ns = [40, 80, 120, 160]
fig, ax = plt.subplots(figsize=(COL, 2.35))
sty = [("RAVA", BLUE, "o", "-", 1.8), ("LOS", DGREY, "s", "-", 1.0), ("SPA", GREY, "^", "--", 1.0), ("Overbook", AMBER, "P", "-", 1.0),
       ("GVMPAC2", "#7a7a7a", "x", ":", 1.0), ("FPA", GREY, "v", ":", 1.0), ("McAfee", LGREY, "D", "-.", 1.0), ("DRF", "#b5b5b5", "<", "--", 1.0), ("OPD", "#d0d0d0", ">", ":", 1.0)]
for k, c, mk, ls, lw in sty:
    ax.errorbar(ns, [OUT["main"][str(n)][k]["eff"] for n in ns], yerr=[OUT["main"][str(n)][k]["eff_ci"] for n in ns], color=c, marker=mk, ms=3.2,
                ls=ls, lw=lw, capsize=1.8, label=NAME[k], zorder=3 if k == "RAVA" else 2)
ax.errorbar([40], [OUT["vcg40"]["eff"]], yerr=[OUT["vcg40"]["ci"]], color=RED, marker="*", ms=7, ls="none", capsize=2, label=r"VCG (exact, $n{=}40$)", zorder=4)
ax.set_xticks(ns); ax.set_xlabel(r"Jobs per epoch, $n$"); ax.set_ylabel(r"Efficiency (\% of LP bound)"); ax.grid(axis="y"); ax.set_ylim(40, 95)
ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.25), fontsize=6.5, columnspacing=0.8, handlelength=1.8)
fig.savefig(FIG + "fig_efficiency.pdf"); plt.close(fig)

# per-instance scatter vs LOS (200 cases)
fig, ax = plt.subplots(figsize=(COL, 2.45)); wins = tot = 0
cols = {"40": "#a9c4ee", "80": "#6f9be0", "120": "#2e6fd0", "160": "#0b3d91"}
for n, runs in M.items():
    x = [r["LOS"]["welfare"] / r["bound"] * 100 for r in runs]; y = [r["RAVA"]["welfare"] / r["bound"] * 100 for r in runs]
    wins += sum(b > a + 1e-9 for a, b in zip(x, y)); tot += len(x)
    ax.scatter(x, y, s=9, color=cols[n], edgecolor="white", lw=0.3, label=f"$n={n}$", zorder=3)
ax.plot([65, 97], [65, 97], "k--", lw=0.7); ax.fill_between([65, 97], [65, 97], 97, color=BLUE, alpha=0.06)
ax.text(67, 93.5, f"RAVA better in {wins} of {tot} workloads", fontsize=7, weight="bold"); ax.text(74, 66.5, "LOS better", fontsize=7, color=DGREY)
ax.set_xlim(65, 97); ax.set_ylim(65, 97); ax.set_aspect("equal"); ax.grid(lw=0.3)
ax.set_xlabel(r"LOS efficiency (\%)"); ax.set_ylabel(r"RAVA efficiency (\%)"); ax.legend(frameon=False, loc="lower right", fontsize=6.3)
fig.savefig(FIG + "fig_scatter.pdf"); plt.close(fig); OUT["scatter"] = dict(wins=int(wins), total=int(tot))

# ablation (n = 120, 160)
var = [("RAVA-no-rho", "w/o\n$g(\\rho)$"), ("RAVA-no-size", "$\\alpha{=}0$\n(no size)"), ("RAVA-a1", "$\\alpha{=}1$"), ("RAVA-no-phi", "$\\lambda{=}0$\n(no flex.)"), ("RAVA-E", "$\\gamma{=}1$\n(eff.)")]
fig, ax = plt.subplots(figsize=(COL, 2.1)); x = np.arange(len(var)); wb = 0.38
for j, (n, c) in enumerate([("120", DGREY), ("160", BLUE)]):
    st = [OUT["main"][n][k] for k, _ in var]
    ax.bar(x + (j - 0.5) * wb, [s["gain"] for s in st], wb, yerr=[[s["gain"] - s["lo"] for s in st], [s["hi"] - s["gain"] for s in st]], color=c, capsize=2, error_kw=dict(lw=0.6), label=f"$n={n}$")
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([l for _, l in var]); ax.grid(axis="y")
ax.set_ylabel(r"Loss vs.\ full RAVA (\%)"); ax.legend(frameon=False)
fig.savefig(FIG + "fig_ablation.pdf"); plt.close(fig)

# radar (n = 120)
runs = M["120"]; mech = [("RAVA", BLUE), ("LOS", DGREY), ("Overbook", AMBER), ("FPA", "#8e44ad"), ("McAfee", RED)]
mets = [("welfare", "Welfare", 1), ("revenue", "Revenue", 1), ("util", "Utilization", 1), ("waste", "Low waste", -1), ("jain", "Fairness", 1), ("served", "Jobs served", 1)]
def val(r, m, k): return (r[m]["reserved"] - r[m]["util"]) if k == "waste" else r[m][k]
arr = np.array([[np.mean([val(r, m, k) for r in runs]) * s for k, _, s in mets] for m, _ in mech]); lo, hi = arr.min(0), arr.max(0); nr = 0.15 + 0.85 * (arr - lo) / (hi - lo)
ang = np.linspace(0, 2 * np.pi, len(mets), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(COL, 2.8)); ax = fig.add_subplot(111, polar=True)
for (m, c), row in zip(mech, nr):
    d = row.tolist() + row[:1].tolist(); ax.plot(ang, d, color=c, lw=2 if m == "RAVA" else 1, label=NAME[m]); ax.fill(ang, d, color=c, alpha=0.18 if m == "RAVA" else 0.04)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([l for _, l, _ in mets], fontsize=7); ax.set_yticklabels([]); ax.set_ylim(0, 1.05); ax.grid(lw=0.4)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=6.8)
fig.savefig(FIG + "fig_radar.pdf"); plt.close(fig)

# ============================================================ deposit / belief study
K = json.load(open("res_kappa.json")); OUT["kappa"] = {}
fig, axs = plt.subplots(1, 3, figsize=(DCOL, 1.85), sharey=True)
for ax, (bel, title) in zip(axs, [("calibrated", "Calibrated beliefs ($\\hat q=q$)"), ("overconfident+0.3", "Overconfident ($\\hat q=q+0.3$)"), ("certain", "Certain of success ($\\hat q=1$)")]):
    ks = [0.0, 0.25, 0.5, 1.0]; g1 = []; g2 = []
    for k in ks:
        rows = [r for r in K if r["belief"] == bel and r["kappa"] == k]
        a = S.paired([r["RAVA"] for r in rows], [r["LOS"] for r in rows]); b = S.paired([r["RAVA-naive"] for r in rows], [r["LOS"] for r in rows])
        g1.append(a); g2.append(b); OUT["kappa"][f"{bel}_{k}"] = dict(rava=a, naive=b)
    ax.fill_between(ks, [s["lo"] for s in g1], [s["hi"] for s in g1], color=BLUE, alpha=0.15, lw=0)
    ax.plot(ks, [s["gain"] for s in g1], color=BLUE, marker="o", ms=3.5, label=r"RAVA, $g(\rho)=\rho+(1-\rho)\kappa$")
    ax.plot(ks, [s["gain"] for s in g2], color=RED, marker="s", ms=3, ls="--", label=r"naive score $g(\rho)=\rho$")
    ax.axhline(0, color="k", lw=0.6); ax.set_title(title); ax.set_xlabel(r"Deposit rate $\kappa$"); ax.set_xticks(ks); ax.grid(axis="y")
axs[0].set_ylabel(r"Welfare gain vs.\ LOS (\%)"); axs[2].legend(frameon=False, fontsize=6.3, loc="lower left")
fig.savefig(FIG + "fig_kappa.pdf"); plt.close(fig)

# ============================================================ unreliable share sweep
B = json.load(open("res_bad.json")); bads = [0.0, 0.2, 0.4, 0.6]; OUT["bad"] = {}
fig, ax = plt.subplots(figsize=(COL, 2.0))
for k, c, mk in [("RAVA", BLUE, "o"), ("Overbook", AMBER, "P"), ("LOS", DGREY, "s"), ("GVMPAC2", GREY, "x")]:
    e = [[r[k] / r["bound"] * 100 for r in B if r["bad"] == b] for b in bads]
    ax.errorbar([b * 100 for b in bads], [np.mean(v) for v in e], yerr=[1.96 * np.std(v, ddof=1) / np.sqrt(len(v)) for v in e], color=c, marker=mk, ms=3.5, capsize=2, label=NAME[k], lw=1.8 if k == "RAVA" else 1)
for b in bads:
    rows = [r for r in B if r["bad"] == b]; OUT["bad"][str(b)] = dict(vs_los=S.paired([r["RAVA"] for r in rows], [r["LOS"] for r in rows]), vs_ob=S.paired([r["RAVA"] for r in rows], [r["Overbook"] for r in rows]))
ax.set_xlabel(r"Share of unreliable tenants (\%)"); ax.set_ylabel(r"Efficiency (\%)"); ax.set_xticks([0, 20, 40, 60]); ax.grid(axis="y"); ax.legend(frameon=False, ncol=2, fontsize=6.5)
fig.savefig(FIG + "fig_bad.pdf"); plt.close(fig)

# ============================================================ adversaries
A = json.load(open("res_adversary.json")); OUT["exploit"] = {}; OUT["whitewash"] = {}
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 1.9), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]
for k, c, ls in [("LOS", DGREY, "-"), ("RAVA z=1", BLUE, "-"), ("RAVA z=0.8", GREEN, "--")]:
    L = np.array([r[1] for r in A["exploit"][k]]) * 100; m = L.mean(0); h = 1.96 * L.std(0, ddof=1) / np.sqrt(len(L))
    lab = "LOS" if k == "LOS" else (r"RAVA, $\zeta=1$" if k == "RAVA z=1" else r"RAVA, $\zeta=0.8$")
    ax.plot(np.arange(1, 31), m, color=c, ls=ls, label=lab, lw=1.6 if "RAVA" in k else 1); ax.fill_between(np.arange(1, 31), m - h, m + h, color=c, alpha=0.12, lw=0)
for k in A["exploit"]:
    L = np.array([r[1] for r in A["exploit"][k]]); W = np.array([r[0] for r in A["exploit"][k]])
    OUT["exploit"][k] = dict(lost_after=float(L[:, 10:].sum(1).mean() * 100), welfare_after=float(W[:, 10:].sum(1).mean()))
OUT["exploit"]["rava_vs_los_lost"] = S.paired(np.array([r[1] for r in A["exploit"]["RAVA z=1"]])[:, 10:].sum(1), np.array([r[1] for r in A["exploit"]["LOS"]])[:, 10:].sum(1))
ax.axvline(10.5, color="k", ls=":", lw=0.8); ax.text(11, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] > 0 else 5, "exploiters start defaulting", fontsize=6.5)
ax.set_xlabel("Round"); ax.set_ylabel(r"Capacity lost to exploiters (\%)"); ax.legend(frameon=False, loc="upper left", fontsize=6.5); ax.grid(axis="y"); ax.set_xlim(1, 30)
ax = axs[1]; keys = [("RAVA no reset", "RAVA,\nno resets"), ("RAVA prior 0.8", "resets,\n$\\rho_0{=}0.8$"), ("RAVA prior 0.5", "resets,\n$\\rho_0{=}0.5$"), ("RAVA prior 0.3", "resets,\n$\\rho_0{=}0.3$"), ("LOS", "LOS")]
ref = np.array([r[0] for r in A["whitewash"]["LOS"]]).sum(1)
for j, (k, lab) in enumerate(keys):
    W = np.array([r[0] for r in A["whitewash"][k]]).sum(1); st = S.paired(W, ref); OUT["whitewash"][k] = st
    ax.bar(j, st["gain"], yerr=[[st["gain"] - st["lo"]], [st["hi"] - st["gain"]]], color=BLUE if j == 0 else (DGREY if k == "LOS" else "#7aa6e6"), capsize=2, error_kw=dict(lw=0.6))
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(range(len(keys))); ax.set_xticklabels([l for _, l in keys], fontsize=6.3); ax.set_ylabel(r"Welfare vs.\ LOS (\%)"); ax.grid(axis="y")
fig.savefig(FIG + "fig_adversary.pdf"); plt.close(fig)

# empirical regret
G = json.load(open("res_regret.json")); OUT["regret"] = {k: dict(n=len(v), max=float(np.max(v)), mean=float(np.mean(v)), pos=int(np.sum(np.array(v) > 1e-6))) for k, v in G.items()}
fig, ax = plt.subplots(figsize=(COL, 1.9)); ks = list(G.keys()); data = [np.array(G[k]) for k in ks]
parts = ax.violinplot([np.clip(d, -40, 5) for d in data], showextrema=False, widths=0.8)
for pc in parts["bodies"]: pc.set_facecolor(BLUE); pc.set_alpha(0.35)
for j, d in enumerate(data):
    ax.scatter(np.full(len(d), j + 1) + np.random.default_rng(j).uniform(-0.18, 0.18, len(d)), np.clip(d, -40, 5), s=1.2, color=DGREY, alpha=0.5)
    ax.text(j + 1, 3.0, f"max {max(d.max(), 0):.0f}", ha="center", fontsize=5.8)
ax.axhline(0, color=RED, lw=0.8); ax.set_xticks(range(1, len(ks) + 1)); ax.set_xticklabels([k.replace("bid x", "bid\n$\\times$").replace(" window", "\nwindow") for k in ks], fontsize=6.2)
ax.set_ylabel("Utility gain from lying"); ax.set_ylim(-42, 6); ax.grid(axis="y")
fig.savefig(FIG + "fig_regret.pdf"); plt.close(fig)
O = np.array(json.load(open("res_overbid.json"))); OUT["overbid"] = dict(mean=float(O.mean()), ci=float(1.96 * O.std(ddof=1) / np.sqrt(len(O))), max=float(O.max()), pos=float((O > 1e-9).mean()), n=len(O))

# ============================================================ scalability
Sc = json.load(open("res_scale.json")); nn = sorted(set(r["n"] for r in Sc)); OUT["scale"] = {}
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 1.95))
for k, c, mk in [("RAVA", BLUE, "o"), ("LOS", DGREY, "s"), ("GVMPAC2", GREY, "x"), ("OPD", LGREY, ">")]:
    t = [np.mean([r[k]["t_total"] for r in Sc if r["n"] == n]) for n in nn]; e = [np.mean([r[k]["welfare"] / r["bound"] * 100 for r in Sc if r["n"] == n]) for n in nn]
    axs[0].plot(nn, t, color=c, marker=mk, ms=3.5, label=NAME[k], lw=1.6 if k == "RAVA" else 1); axs[1].plot(nn, e, color=c, marker=mk, ms=3.5, label=NAME[k], lw=1.6 if k == "RAVA" else 1)
    OUT["scale"][k] = dict(t=dict(zip(map(str, nn), t)), eff=dict(zip(map(str, nn), e)))
ta = [np.mean([r["RAVA"]["t_alloc"] for r in Sc if r["n"] == n]) for n in nn]; tl = [np.mean([r["t_lp"] for r in Sc if r["n"] == n]) for n in nn]
axs[0].plot(nn, ta, color=BLUE, ls=":", marker=".", label="RAVA allocation only"); axs[0].plot(nn, tl, color=GREEN, ls="--", label="LP bound (HiGHS)")
axs[0].scatter([40], [OUT["vcg40"]["t"]], color=RED, marker="*", s=40, zorder=5, label=r"exact VCG ($n{=}40$)")
OUT["scale"]["RAVA"]["t_alloc"] = dict(zip(map(str, nn), ta)); OUT["scale"]["lp"] = dict(zip(map(str, nn), tl))
axs[0].set_xlim(30, 13000); axs[1].set_xlim(130, 13000); axs[0].set_xscale("log"); axs[0].set_yscale("log"); axs[0].set_xlabel("Jobs per epoch $n$"); axs[0].set_ylabel("Time per auction (s)"); axs[0].legend(frameon=False, fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(1.1, -0.25)); axs[0].grid(lw=0.3)
axs[1].set_xscale("log"); axs[1].set_xlabel("Jobs per epoch $n$ (capacity scaled with $n$)"); axs[1].set_ylabel(r"Efficiency (\%)"); axs[1].grid(axis="y"); axs[1].set_ylim(40, 95)
fig.savefig(FIG + "fig_scale.pdf"); plt.close(fig)
OUT["scale"]["gain10k"] = S.paired([r["RAVA"]["welfare"] for r in Sc if r["n"] >= 2000], [r["LOS"]["welfare"] for r in Sc if r["n"] >= 2000])

# ============================================================ flexibility disclosure
F = json.load(open("res_flex.json")); lams = ["0.0", "0.1", "0.2", "0.4", "0.8"]; OUT["flex"] = {}
ref = [r["welfare"] for r in F if r["lam"] == "none"]
fig, ax = plt.subplots(figsize=(COL, 2.0)); ax2 = ax.twinx()
g = []; dsc = []
for l in lams:
    rows = [r for r in F if r["lam"] == l]; st = S.paired([r["welfare"] for r in rows], ref); g.append(st); dsc.append(np.mean([r["disclosed"] for r in rows]) * 100)
    OUT["flex"][l] = dict(gain=st, disclosed=dsc[-1])
full = S.paired([r["welfare"] for r in F if r["lam"] == "full"], ref); OUT["flex"]["full"] = full
xs = [float(l) for l in lams]
ax.fill_between(xs, [s["lo"] for s in g], [s["hi"] for s in g], color=BLUE, alpha=0.15, lw=0); ax.plot(xs, [s["gain"] for s in g], color=BLUE, marker="o", ms=3.5, label="welfare gain")
ax.axhline(full["gain"], color=GREEN, ls="--", lw=0.9); ax.text(0.45, full["gain"] + 0.25, "everyone discloses (upper reference)", fontsize=6.3, color=GREEN)
ax2.bar(xs, dsc, width=0.05, color=GREY, alpha=0.45, label="tenants disclosing slack"); ax2.set_ylim(0, 260); ax2.set_ylabel(r"Tenants disclosing (\%)", color=DGREY)
ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)
ax.set_xlabel(r"Flexibility bonus $\lambda$"); ax.set_ylabel(r"Welfare gain vs.\ no disclosure (\%)"); ax.grid(axis="y"); ax.set_xticks(xs)
fig.savefig(FIG + "fig_flex.pdf"); plt.close(fig)

# ============================================================ Philly
R = json.load(open("res_philly_1-2.json")) + json.load(open("res_philly_3.json"))
OB = {(r["day"], r["seed"]): r["Overbook"] for r in json.load(open("res_philly_overbook.json")) if "killedok" not in r["file"]}
CB = {(r["day"], r["seed"]): r for r in json.load(open("res_philly_combo.json"))}
OPDT = {(r["day"], r["seed"]): r["welfare"] for r in json.load(open("res_opd_tuned.json"))["philly"]["test"]}
for r in R:
    r["Overbook"] = OB[(r["day"], r["seed"])]
    for k in ["RAVA+OB0.1", "RAVA+OB0.2", "LOS+OB0.2"]: r[k] = CB[(r["day"], r["seed"])][k]
    r["OPD-tuned"] = {"welfare": OPDT[(r["day"], r["seed"])]}
Tt = [r for r in R if r["day"] >= "2017-10-17"]; Vv = [r for r in R if r["day"] < "2017-10-17"]
grid = {}
for r in Vv:
    for k, v in r["grid"].items(): grid.setdefault(k, []).append(v["welfare"] / r["LOS"]["welfare"])
best = max(grid, key=lambda k: np.mean(grid[k]))
for r in Tt: r["RAVA-tuned"] = r["grid"][best]
pm = ["FPA", "SPA", "LOS", "GVMPAC2", "OPD", "McAfee", "DRF", "Overbook", "RAVA-no-rho", "RAVA-a1", "RAVA-E", "RAVA-tuned", "LOS+OB0.2", "RAVA+OB0.1", "RAVA+OB0.2"]
OUT["philly"] = dict(n_test=len(Tt), n_val=len(Vv), days_test=len(set(r["day"] for r in Tt)), mean_jobs=float(np.mean([r["n"] for r in Tt])), best=best, tab={})
ps = []; ks = []
for m in pm:
    st = S.paired([r["RAVA"]["welfare"] for r in Tt], [r[m]["welfare"] for r in Tt]); OUT["philly"]["tab"][m] = st; ps.append(st["p_wilcoxon"]); ks.append(m)
for k, a in zip(ks, S.holm(ps)): OUT["philly"]["tab"][k]["p_holm"] = float(a)
for m in pm + ["RAVA"]:
    d = OUT["philly"]["tab"].setdefault(m, {}); e = [r[m]["welfare"] / r["bound"] * 100 for r in Tt]
    d.update(eff=float(np.mean(e)), eff_ci=float(1.96 * np.std(e, ddof=1) / np.sqrt(len(e))))
    for f in ["useful_gpu", "lost_gpu", "reserved_gpu", "revenue", "served", "jain"]:
        d[f] = float(np.mean([r[m][f] for r in Tt])) if f in r[m] else None
OUT["philly"]["a1_vs_gvm"] = S.paired([r["RAVA-a1"]["welfare"] for r in Tt], [r["GVMPAC2"]["welfare"] for r in Tt])
OUT["philly"]["combo_vs_losob"] = S.paired([r["RAVA+OB0.2"]["welfare"] for r in Tt], [r["LOS+OB0.2"]["welfare"] for r in Tt])
OUT["philly"]["combo_vs_rava"] = S.paired([r["RAVA+OB0.2"]["welfare"] for r in Tt], [r["RAVA"]["welfare"] for r in Tt])
OUT["philly"]["evicted"] = float(np.mean([r["Overbook"]["evicted"] for r in Tt]))
KO = [r for r in json.load(open("res_philly_killedok_1.json")) if r["day"] >= "2017-10-17"]
OUT["philly"]["killed_ok"] = S.paired([r["RAVA"]["welfare"] for r in KO], [r["LOS"]["welfare"] for r in KO])
OUT["philly"]["gamma"] = {g: dict(eff=float(np.mean([r["grid"][f"0.5_{g}"]["welfare"] / r["bound"] for r in Tt]) * 100),
                                  useful=float(np.mean([r["grid"][f"0.5_{g}"]["useful_gpu"] for r in Tt]) * 100)) for g in ["0.0", "0.5", "1.0"]}

order = ["DRF", "OPD", "McAfee", "FPA", "SPA", "GVMPAC2", "LOS", "RAVA-no-rho", "RAVA-a1", "Overbook", "RAVA", "LOS+OB0.2", "RAVA+OB0.2"]
lab = dict(NAME, **{"LOS+OB0.2": "LOS + overbook 20\\%", "RAVA+OB0.2": "RAVA + overbook 20\\%"})
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.5), gridspec_kw=dict(width_ratios=[1.05, 1], wspace=0.55))
ax = axs[0]; y = np.arange(len(order))
e = [OUT["philly"]["tab"][m]["eff"] for m in order]; ci = [OUT["philly"]["tab"][m]["eff_ci"] for m in order]
cl = [BLUE if m.startswith("RAVA") and m != "RAVA-no-rho" else (AMBER if "OB" in m or m == "Overbook" else GREY) for m in order]
ax.barh(y, e, xerr=ci, color=cl, height=0.62, capsize=1.5, error_kw=dict(lw=0.6))
for i, (v, c_) in enumerate(zip(e, ci)): ax.text(v + c_ + 0.8, i, f"{v:.1f}", va="center", fontsize=6.3)
ax.set_yticks(y); ax.set_yticklabels([lab[m] for m in order], fontsize=6.6); ax.set_xlim(60, 98); ax.set_xlabel(r"Efficiency on held-out days (\% of oracle bound)"); ax.grid(axis="x")
ax = axs[1]; o2 = ["LOS", "GVMPAC2", "Overbook", "RAVA", "RAVA-E", "RAVA+OB0.2"]
U = np.array([OUT["philly"]["tab"][m]["useful_gpu"] * 100 for m in o2]); Lo = np.array([OUT["philly"]["tab"][m]["lost_gpu"] * 100 for m in o2])
Rs = np.array([OUT["philly"]["tab"][m]["reserved_gpu"] * 100 for m in o2]); I = Rs - Lo - U
ax.barh(range(len(o2)), U, color=BLUE, label="useful GPU work", height=0.6); ax.barh(range(len(o2)), I, left=U, color="#9fbbe8", label="reserved, idle", height=0.6)
ax.barh(range(len(o2)), Lo, left=U + I, color=RED, label="held by jobs that failed", height=0.6)
ax.set_yticks(range(len(o2))); ax.set_yticklabels([lab[m] for m in o2], fontsize=6.6); ax.set_xlabel(r"Share of GPU-hours (\%)"); ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.22), ncol=3)
fig.savefig(FIG + "fig_philly.pdf"); plt.close(fig)

# ============================================================ live Slurm run
L = json.load(open("../slurm/ledger.json")); Sx = {b["name"]: b for b in json.load(open("../slurm/settlement.json"))}
comp = {}
for line in open("../slurm/jobcomp.log"):
    f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv); comp[f["JobId"]] = f
import datetime as dt
t0 = dt.datetime.fromisoformat(L["epoch_start"]); fig, ax = plt.subplots(figsize=(COL, 1.9)); rows = [j for j in L["jobs"] if j["won"]]
OUT["slurm"] = dict(delays=[], auction_ms=L["auction_seconds"] * 1000)
for k, j in enumerate(rows):
    f = comp[j["slurm_id"]]; s = (dt.datetime.fromisoformat(f["StartTime"]) - t0).total_seconds(); e2 = (dt.datetime.fromisoformat(f["EndTime"]) - t0).total_seconds()
    ps_ = j["planned_slot"] * L["slot"]; ln = {"J2": 2, "J3": 2, "J4": 2, "J5": 2, "J6": 4}[j["name"]] * L["slot"]
    ax.barh(k, ln, left=ps_, color="none", edgecolor=DGREY, ls="--", lw=0.8, height=0.7)
    ok = f["JobState"] == "COMPLETED"; ax.barh(k, e2 - s, left=s, color=BLUE if ok else RED, height=0.45)
    ax.text(max(e2, ps_ + ln) + 2, k, f"{f['JobState'].lower()}, start {s - ps_:+.0f}\\,s", va="center", fontsize=6.2)
    OUT["slurm"]["delays"].append(s - ps_)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{j['name']} ({j['user']})" for j in rows], fontsize=6.6); ax.invert_yaxis()
for t in range(0, 5): ax.axvline(t * L["slot"], color=LGREY, lw=0.5)
ax.set_xlabel(r"Seconds after epoch start (slot $=$ 20\,s; dashed: reserved slot, solid: actual run)"); ax.set_xlim(0, 125)
fig.savefig(FIG + "fig_slurm.pdf"); plt.close(fig)
OUT["slurm"]["settlement"] = list(Sx.values())

json.dump(OUT, open("summary.json", "w"), indent=1, default=float)
print("ok; philly best config", best)
