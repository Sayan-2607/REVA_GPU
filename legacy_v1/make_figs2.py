import json, numpy as np, matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
exec(open("make_figs.py").read().split("COL, DCOL")[0])          # same rcParams as the other figures
COL, DCOL = 3.5, 7.16
BLUE, GREY, DGREY, RED, GREEN, AMBER = "#1f5fbf", "#9a9a9a", "#555555", "#c0392b", "#2e8b57", "#e0a030"
OUT = "legacy_figures/"
fast = json.load(open("results_fast.json")); ex = json.load(open("results_extra.json"))

# ---------- (a) slope chart: rank by bid -> rank by RAVA score (worked example)
jobs = ["J1 LLM fine-tune", "J2 Vision training", "J3 Inference burst", "J4 Hyper-param. sweep", "J5 CPU preprocessing", "J6 Student RL"]
bid = [130, 48, 30, 40, 18, 22]; score = [34.00, 35.47, 29.40, 28.00, 13.58, 14.31]; rho = [0.55, 0.95, 0.98, 0.90, 0.97, 0.92]
won = [False, True, True, True, True, True]
rb = np.argsort(np.argsort(-np.array(bid))) + 1; rs = np.argsort(np.argsort(-np.array(score))) + 1
fig, ax = plt.subplots(figsize=(COL, 2.45))
for i in range(6):
    c = RED if not won[i] else BLUE
    ax.plot([0, 1], [rb[i], rs[i]], color=c, lw=2.2 if i in (0, 1) else 1.2, alpha=0.95, marker="o", ms=5)
    ax.text(-0.04, rb[i], f"{jobs[i]}  ({bid[i]})", ha="right", va="center", fontsize=6.6)
    ax.text(1.04, rs[i], f"{score[i]:.1f}  $\\rho$={rho[i]:.2f}" + ("" if won[i] else "  rejected"), ha="left", va="center", fontsize=6.6, color=c)
ax.set_xlim(-1.25, 1.75); ax.set_ylim(6.6, 0.4); ax.axis("off")
ax.text(0, 0.0, "Rank by bid", ha="center", fontsize=7.5, weight="bold"); ax.text(1, 0.0, "Rank by RAVA score", ha="center", fontsize=7.5, weight="bold")
fig.savefig(OUT + "fig_rank.pdf"); plt.close(fig)

# ---------- (b) utility vs bid: why bidding the true value is best (J2)
v, c = 48.0, 46.02
b = np.linspace(0, 80, 400); u = np.where(b >= c, v - c, 0.0)
fig, ax = plt.subplots(figsize=(COL, 1.95))
ax.axvspan(0, c, color=GREY, alpha=0.12, lw=0); ax.axvspan(c, 80, color=BLUE, alpha=0.08, lw=0)
ax.plot(b, u, color=BLUE, lw=2)
ax.axvline(c, color=DGREY, ls="--", lw=0.8); ax.axvline(v, color=GREEN, lw=1.2)
ax.text(c / 2, 2.6, "bid too low:\nlose, pay 0", ha="center", fontsize=7, color=DGREY)
ax.text(66, 2.6, "win, always pay\nthe critical price", ha="center", fontsize=7, color=BLUE)
ax.annotate(f"critical price\n$c$ = {c:.2f}", (c, 0.2), xytext=(18, 1.25), fontsize=6.8, arrowprops=dict(arrowstyle="->", lw=0.6))
ax.annotate(f"true value $v$ = {v:.0f}\n(best bid)", (v, v - c), xytext=(56, 1.05), fontsize=6.8, color=GREEN, arrowprops=dict(arrowstyle="->", lw=0.6, color=GREEN))
ax.set_xlabel("Bid of J2"); ax.set_ylabel("Utility of J2"); ax.set_ylim(-0.3, 4); ax.set_xlim(0, 80)
fig.savefig(OUT + "fig_truthful.pdf"); plt.close(fig)

# ---------- (c) where the GPUs go over one day (n=120, mean of 20 workloads)
pr = ex["profile"]; t = np.arange(24)
fig, axs = plt.subplots(1, 3, figsize=(DCOL, 1.95), sharey=True)
for ax, (k, title) in zip(axs, [("LOS", "LOS combinatorial"), ("RAVA", r"RAVA ($\gamma=0$)"), ("RAVA-E", r"RAVA-E ($\gamma=1$)")]):
    p = pr[k]; U, I, L = np.array(p["used"]), np.array(p["idle"]), np.array(p["lost"])
    ax.stackplot(t, U, I, L, colors=[BLUE, "#9fbbe8", RED], alpha=0.92, step=None,
                 labels=["used by running jobs", "reserved but idle", "lost to no-shows"])
    ax.axhline(32, color="k", ls="--", lw=0.7)
    ax.set_title(f"{title}: GPUs used {U.sum()/(32*24)*100:.0f}\\%, idle {I.sum()/(32*24)*100:.0f}\\%, lost {L.sum()/(32*24)*100:.0f}\\%") if False else ax.set_title(f"{title}: used {U.sum()/(32*24)*100:.0f}\\%, lost {L.sum()/(32*24)*100:.0f}\\%")
    ax.set_xlim(0, 23); ax.set_ylim(0, 35); ax.set_xticks([0, 6, 12, 18, 23]); ax.set_xticklabels(["0h", "6h", "12h", "18h", "23h"])
axs[0].set_ylabel("Expected GPUs (of 32)")
axs[0].legend(frameon=False, loc="lower center", ncol=1, fontsize=6.3)
fig.savefig(OUT + "fig_gpu_day.pdf"); plt.close(fig)

# ---------- (d) alpha x gamma heatmaps
gr = ex["grid"]; A, G = gr["A"], gr["G"]; E, Ut = np.array(gr["eff"]), np.array(gr["util"])
fig, axs = plt.subplots(1, 2, figsize=(DCOL, 2.25), gridspec_kw=dict(wspace=0.42))
for ax, M, title, cmap in [(axs[0], E, "Efficiency (\\% of LP bound)", "Blues"), (axs[1], Ut, "Effective utilization (\\%)", "Oranges")]:
    im = ax.imshow(M, cmap=cmap, origin="lower", aspect="auto")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=6.8,
                    color="white" if (M[i, j] - M.min()) / (M.max() - M.min()) > 0.6 else "black")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{g:g}" for g in G]); ax.set_yticks(range(5)); ax.set_yticklabels([f"{a:g}" for a in A])
    ax.set_xlabel(r"Efficiency weight $\gamma$"); ax.set_ylabel(r"Size weight $\alpha$"); ax.set_title(title)
    ax.spines[:].set_visible(False); fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    k = np.unravel_index(np.argmax(M), M.shape); ax.add_patch(plt.Rectangle((k[1] - 0.5, k[0] - 0.5), 1, 1, fill=False, ec="k", lw=1.4))
fig.savefig(OUT + "fig_heatmap.pdf"); plt.close(fig)

# ---------- (e) radar chart at n=120
runs = fast["120"]
mech = [("RAVA", "RAVA", BLUE), ("CA-LOS", "LOS", DGREY), ("SPA", "Second-price", AMBER), ("FPA", "First-price", "#8e44ad"), ("McAfee-DA", "McAfee DA", RED)]
mets = [("welfare", "Welfare", 1), ("revenue", "Revenue", 1), ("util", "Utilization", 1), ("waste", "Low waste", -1), ("jain", "Fairness", 1), ("accepted", "Jobs served", 1)]
val = {k: [np.mean([r[k][m] for r in runs]) for m, _, _ in mets] for k, _, _ in mech}
arr = np.array([val[k] for k, _, _ in mech]); sgn = np.array([s for _, _, s in mets])
arr = arr * sgn; lo, hi = arr.min(0), arr.max(0); nr = 0.15 + 0.85 * (arr - lo) / (hi - lo)
ang = np.linspace(0, 2 * np.pi, len(mets), endpoint=False).tolist(); ang += ang[:1]
fig = plt.figure(figsize=(COL, 2.9)); ax = fig.add_subplot(111, polar=True)
for (k, lab, c), row in zip(mech, nr):
    d = row.tolist() + row[:1].tolist()
    ax.plot(ang, d, color=c, lw=2.0 if k == "RAVA" else 1.0, label=lab, zorder=3 if k == "RAVA" else 2)
    ax.fill(ang, d, color=c, alpha=0.18 if k == "RAVA" else 0.04)
ax.set_xticks(ang[:-1]); ax.set_xticklabels([l for _, l, _ in mets], fontsize=7); ax.set_yticklabels([]); ax.set_ylim(0, 1.05)
ax.spines["polar"].set_linewidth(0.4); ax.grid(lw=0.4)
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3, fontsize=6.8)
fig.savefig(OUT + "fig_radar.pdf"); plt.close(fig)

# ---------- (f) every test case: RAVA vs LOS
fig, ax = plt.subplots(figsize=(COL, 2.5))
cols = {"40": "#a9c4ee", "80": "#6f9be0", "120": "#2e6fd0", "160": "#0b3d91"}; wins = 0; tot = 0
for n, runs in fast.items():
    x = [r["CA-LOS"]["welfare"] / r["bound"] * 100 for r in runs]; y = [r["RAVA"]["welfare"] / r["bound"] * 100 for r in runs]
    wins += sum(yy > xx for xx, yy in zip(x, y)); tot += len(x)
    ax.scatter(x, y, s=13, color=cols[n], edgecolor="white", lw=0.4, label=f"$n={n}$", zorder=3)
ax.plot([70, 95], [70, 95], color="k", lw=0.7, ls="--"); ax.fill_between([70, 95], [70, 95], 95, color=BLUE, alpha=0.06)
ax.text(73, 92, "RAVA better", color=BLUE, fontsize=7); ax.text(79, 71.5, "LOS better", color=DGREY, fontsize=7)
ax.text(73, 89.6, f"RAVA wins {wins} of {tot} cases", fontsize=7, weight="bold")
ax.set_xlim(70, 95); ax.set_ylim(70, 95); ax.set_aspect("equal")
ax.set_xlabel(r"LOS efficiency (\% of bound)"); ax.set_ylabel(r"RAVA efficiency (\% of bound)")
ax.legend(frameon=False, loc="lower right", fontsize=6.5); ax.grid(lw=0.3)
fig.savefig(OUT + "fig_scatter.pdf"); plt.close(fig)
print("wins", wins, tot)
