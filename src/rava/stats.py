"""Paired statistics used throughout the paper."""
import numpy as np
from scipy import stats

def paired(x, y, n_boot=10000, seed=0):
    """Compare treatment x against baseline y on the same instances.
    Returns mean % gain (mean of per-instance ratios), bootstrap 95% CI of that mean, Wilcoxon signed-rank p,
    paired t-test p, Cohen's d_z on the differences, and wins/ties/losses."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    g = (x / y - 1) * 100; d = x - y
    rng = np.random.default_rng(seed)
    bs = g[rng.integers(0, len(g), (n_boot, len(g)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    nz = d[np.abs(d) > 1e-9]
    pw = stats.wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
    pt = stats.ttest_rel(x, y).pvalue if np.std(d) > 0 else 1.0
    dz = d.mean() / d.std(ddof=1) if d.std(ddof=1) > 0 else 0.0
    return dict(gain=float(g.mean()), lo=float(lo), hi=float(hi), p_wilcoxon=float(pw), p_t=float(pt), dz=float(dz),
                wins=int((d > 1e-9).sum()), ties=int((np.abs(d) <= 1e-9).sum()), losses=int((d < -1e-9).sum()), k=len(g))

def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    p = np.asarray(pvals, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (m - r) * p[i])); adj[i] = run
    return adj

def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"
