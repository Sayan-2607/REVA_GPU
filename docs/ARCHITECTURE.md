# Where each part of the paper lives in the code

| Paper | Code |
|---|---|
| Sec. III, Eq. (1)–(2): servers, jobs, capacity, footprint | `rava/model.py` (`Jobs`, `footprint`, `jobs_from_records`) |
| Def. 1, Eq. (3), Lemma 1: settlement rule, effective value | `rava/model.py` (`effective_value`, `expected_utility`), `rava/settlement.py` |
| Eq. (5): Beta reputation, forgetting ζ, floor ρ_min, newcomer prior | `rava/reputation.py` (`BetaReputation`) |
| Eq. (7): score, g_κ(ρ) | `rava/mechanism.py` (`rava_weight`, `g_kappa`) |
| Algorithm 1: earliest-fit, best-fit servers, gang jobs | `rava/placement.py` (`try_place`, `place`) — Numba |
| Algorithm 2: critical prices by early-exit scan | `rava/placement.py` (`critical_other`), `rava/mechanism.py` (`greedy`) |
| Sec. IV-D: reserve adaptation, early-release credit | `rava/settlement.py` |
| Lemma 2, Theorem 1 (monotonicity, DSIC, IR) | verified in `tests/test_mechanism.py` |
| Prop. 1, Cor. 1, Lemma 3, Theorem 2, Prop. 4–5 | `rava/theory.py`; verified in `tests/test_theory.py` |
| Prop. 3 (special cases LOS / second price) | `tests/test_mechanism.py::test_rava_reduces_to_los_and_second_price` |
| Sec. VI: worked example | `rava/workloads/example.py`, `rava example`, `tests/test_example.py` |
| Sec. VII: Slurm broker | `rava/integrations/slurm_broker.py`, `rava slurm ...`, `slurm/` |
| Sec. VIII-A baselines | `rava/baselines.py` (first/second price, LOS, G-VMPAC-II, OPD, DRF, overbooking, McAfee, exact VCG) |
| Efficiency bound | `rava/bounds.py` (`lp_bound`) |
| Metrics, statistics | `rava/metrics.py`, `rava/stats.py` |
| Sec. VIII-B synthetic experiments | `rava/experiments/synthetic.py` (`main kappa bad flex regret adversary overbid scale`) |
| Sec. VIII-C Philly trace | `rava/experiments/philly.py`, `philly_overbook.py`, `combo.py`, `tune_opd.py`, `philly_util.py`, `prepare_philly.py` |
| Every number and figure | `rava/analysis/figures.py` → `results/summary.json`, `paper/figures/` |
| Figures of Secs. VI and VIII-G (rank, example, GPU-day, heat map) and exact VCG | `legacy_v1/` (first-version code; same workloads) |
