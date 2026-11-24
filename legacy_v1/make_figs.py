import json, numpy as np, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["TeX Gyre Termes", "Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "text.usetex": True, "text.latex.preamble": r"\usepackage{mathptmx}\usepackage{amssymb}", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6,
    "grid.linewidth": 0.4, "grid.alpha": 0.5, "lines.linewidth": 1.2, 
    "axes.spines.top": False, "axes.spines.right": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
COL, DCOL = 3.5, 7.16
BLUE, GREY, DGREY, RED, GREEN = "#1f5fbf", "#9a9a9a", "#555555", "#c0392b", "#2e8b57"
OUT = "legacy_figures/"
fast = json.load(open("results_fast.json"))
ci = lambda a: 1.96 * np.std(a, ddof=1) / np.sqrt(len(a))

# ---------------- Fig: efficiency vs n
mechs = [("RAVA", "RAVA (proposed)", BLUE, "o", "-"), ("CA-LOS", "LOS combinatorial", DGREY, "s", "-"),
         ("SPA", "Second-price", GREY, "^", "--"), ("FPA", "First-price", GREY, "v", ":"),
         ("McAfee-DA", "McAfee double auction", "#b0b0b0", "D", "-.")]
ns = [40, 80, 120, 160]
fig, ax = plt.subplots(figsize=(COL, 2.2))
for k, lab, c, mk, ls in mechs:
    m = [np.mean([r[k]["welfare"] / r["bound"] * 100 for r in fast[str(n)]]) for n in ns]
    e = [ci([r[k]["welfare"] / r["bound"] * 100 for r in fast[str(n)]]) for n in ns]
    ax.errorbar(ns, m, yerr=e, label=lab, color=c, marker=mk, ms=3.5, ls=ls, capsize=2, lw=1.6 if k == "RAVA" else 1.0, zorder=3 if k == "RAVA" else 2)
ax.set_xlabel("Number of jobs bidding per epoch, $n$"); ax.set_ylabel(r"Efficiency (\% of LP bound)")
vcg = json.load(open("results_vcg.json"))
ve = [vcg[s]["VCG"]["welfare"] / fast["40"][s]["bound"] * 100 for s in range(len(vcg))]
ax.errorbar([40], [np.mean(ve)], yerr=[ci(ve)], color=RED, marker="*", ms=7, ls="none", capsize=2, label=r"VCG (exact, $n{=}40$ only)", zorder=4)
ax.set_xticks(ns); ax.grid(axis="y"); ax.set_ylim(64, 94)
ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.27), handlelength=2.2, columnspacing=1.0)
fig.savefig(OUT + "fig_efficiency.pdf"); plt.close(fig)

# ---------------- Fig: gamma trade-off
sw = json.load(open("results_sweep.json"))
g = sorted(float(k) for k in sw); W = [sw[str(k) if str(k) in sw else repr(k)][0] for k in g]
U = [sw[str(k) if str(k) in sw else repr(k)][1] * 100 for k in g]
fig, ax = plt.subplots(figsize=(COL, 2.1))
ax.plot(U, W, color=BLUE, marker="o", ms=4)
for gi, u, w in zip(g, U, W):
    ax.annotate(rf"$\gamma={gi:g}$", (u, w), textcoords="offset points", xytext=(5, 4), fontsize=7)
ax.set_xlabel(r"Effective utilisation (\%)"); ax.set_ylabel(r"Expected realised welfare $\mathbb{E}[SW]$")
ax.grid(True); ax.set_xlim(49, 56.3)
fig.savefig(OUT + "fig_gamma.pdf"); plt.close(fig)

# ---------------- Fig: ablation
var = [("RAVA", "Full\nRAVA"), ("RAVA-no-rho", "w/o\n$\\rho$"), ("RAVA-no-phi", "$\\lambda{=}0$\n(no flex.)"),
       ("RAVA-no-size", "$\\alpha{=}0$\n(no size)"), ("RAVA-E", "$\\gamma{=}1$\n(eff. mode)")]
fig, ax = plt.subplots(figsize=(COL, 2.2))
x = np.arange(len(var)); wbar = 0.38
for j, (n, c) in enumerate([(120, DGREY), (160, BLUE)]):
    runs = fast[str(n)]
    vals = [[(r[k]["welfare"] / r["CA-LOS"]["welfare"] - 1) * 100 for r in runs] for k, _ in var]
    ax.bar(x + (j - 0.5) * wbar, [np.mean(v) for v in vals], wbar, yerr=[ci(v) for v in vals], color=c,
           capsize=2, error_kw=dict(lw=0.6), label=f"$n={n}$", alpha=0.9)
ax.axhline(0, color="k", lw=0.6); ax.set_xticks(x); ax.set_xticklabels([l for _, l in var])
ax.set_ylabel(r"Welfare gain vs.\ LOS (\%)"); ax.legend(frameon=False); ax.grid(axis="y")
fig.savefig(OUT + "fig_ablation.pdf"); plt.close(fig)

# ---------------- Fig: multi-epoch
fig, ax = plt.subplots(figsize=(COL, 2.1))
for fn, lab, c, off in [("results_epoch.json", r"20\% unreliable", GREY, -0.12), ("results_epoch_0.4.json", r"40\% unreliable", BLUE, 0.12)]:
    e = json.load(open(fn)); R = np.array([z["RAVA"] for z in e]); L = np.array([z["CA-LOS"] for z in e])
    gl = (R / L - 1) * 100; m = gl.mean(0); h = 1.96 * gl.std(0, ddof=1) / np.sqrt(len(gl)); ep = np.arange(1, 21)
    ax.fill_between(ep, m - h, m + h, color=c, alpha=0.15, lw=0)
    ax.plot(ep, m, color=c, marker="o", ms=2.5, label=lab)
ax.axhline(0, color="k", lw=0.6); ax.set_xlabel("Epoch"); ax.set_ylabel(r"Realised welfare gain vs.\ LOS (\%)")
ax.set_xticks([1, 5, 10, 15, 20]); ax.grid(axis="y"); ax.legend(frameon=False, loc="upper left")
fig.savefig(OUT + "fig_epochs.pdf"); plt.close(fig)

# ---------------- Fig: runtime (n = 40, log)
vcg = json.load(open("results_vcg.json")); k = len(vcg)
rt = [("First-price", [fast["40"][s]["FPA"]["t"] for s in range(k)]),
      ("McAfee DA", [fast["40"][s]["McAfee-DA"]["t"] for s in range(k)]),
      ("Second-price", [fast["40"][s]["SPA"]["t"] for s in range(k)]),
      ("LOS", [fast["40"][s]["CA-LOS"]["t"] for s in range(k)]),
      ("RAVA", [fast["40"][s]["RAVA"]["t"] for s in range(k)]),
      ("VCG (exact)", [v["VCG"]["t"] for v in vcg])]
fig, ax = plt.subplots(figsize=(COL, 1.9))
vals = [max(np.mean(t), 1e-4) for _, t in rt]
cols = [GREY, GREY, GREY, DGREY, BLUE, RED]
ax.barh(range(len(rt)), vals, color=cols, height=0.6)
for i, v in enumerate(vals): ax.text(v * 1.3, i, f"{v:.3g}\\,s", va="center", fontsize=6.5)
ax.set_xscale("log"); ax.set_yticks(range(len(rt))); ax.set_yticklabels([n for n, _ in rt])
ax.set_xlabel("Mean wall-clock time per auction incl. payments (s), $n=40$"); ax.set_xlim(5e-5, 4e3)
ax.grid(axis="x")
fig.savefig(OUT + "fig_runtime.pdf"); plt.close(fig)

# ---------------- Fig: worked example schedules
jobs = {"J1": 6, "J2": 4, "J3": 2, "J4": 4, "J5": 0, "J6": 2}
sched = {"VCG / second-price / LOS": {"J1": (1, 3), "J3": (2, 2)},
         "RAVA": {"J2": (1, 2), "J3": (2, 2), "J4": (3, 2), "J6": (1, 4)}}
colors = {"J1": RED, "J2": "#4a7fd4", "J3": "#7aa6e6", "J4": "#2e5aa8", "J6": "#a9c4ee"}
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 1.75), sharey=True)
for ax, (title, sc) in zip(axs, sched.items()):
    for t in range(1, 5):
        y = 0
        for j, (s, L) in sc.items():
            if s <= t < s + L and jobs[j] > 0:
                ax.add_patch(Rectangle((t - 0.45, y), 0.9, jobs[j], color=colors[j], ec="white", lw=0.8))
                ax.text(t, y + jobs[j] / 2, j, ha="center", va="center", fontsize=7, color="white", weight="bold")
                y += jobs[j]
    ax.axhline(8, color="k", lw=0.7, ls="--"); ax.text(4.45, 8.15, "capacity = 8 GPUs", ha="right", fontsize=6.5)
    ax.set_xlim(0.4, 4.6); ax.set_ylim(0, 9.2); ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(["18:00", "19:00", "20:00", "21:00"])
    ew = {"VCG / second-price / LOS": 100.9, "RAVA": 148.7}[title]
    ax.set_title(f"{title}:  $\\mathbb{{E}}[SW]={ew}$")
axs[0].set_ylabel("GPUs reserved")
fig.savefig(OUT + "fig_example.pdf"); plt.close(fig)
print("figures written")
