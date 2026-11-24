# RAVA — Reciprocal Adaptive Value and Resource Allocation Auction

**A reliability-aware auction mechanism for shared GPU and cloud clusters.**

RAVA is a research implementation of the paper **“RAVA: A Reciprocal Adaptive Value and Resource Allocation Auction for Reliability-Aware GPU and Cloud Clusters.”** It introduces a reliability-aware resource allocation mechanism designed for shared GPU/CPU clusters, incorporating tenant reliability, resource demand, scheduling constraints, and payment incentives.

The implementation includes the auction mechanism, critical-bid pricing, pay-on-use settlement, adaptive reputation tracking, baseline mechanisms, theoretical analysis, simulation experiments, trace-driven evaluation, and a Slurm integration.

---

## Overview

In shared computing clusters, resource allocation is complicated by heterogeneous workloads, resource contention, scheduling constraints, and the uncertainty that submitted jobs may not execute.

RAVA incorporates tenant reliability into allocation and pricing by penalizing the risk associated with unsuccessful job execution while maintaining a value-based allocation framework.

Jobs are ranked using a scoring function based on bid value, reliability, resource efficiency, deadline-related factors, and demand. The mechanism supports multi-resource allocation, including multi-server gang jobs, and charges winners their critical bids.

### Key Features

- **Reliability-aware allocation** — incorporates tenant execution reliability into auction ranking.
- **Adaptive reputation** — models tenant reliability using a Beta reputation system updated between epochs.
- **Critical-bid pricing** — determines winner payments using allocation-critical bid thresholds.
- **Pay-on-use settlement** — charges based on execution, with forfeitable deposits governed by \(\kappa\).
- **Multi-resource scheduling** — supports GPU, CPU, memory, time windows, and multi-server gang jobs.
- **Mechanism comparisons** — includes auction baselines, scheduling policies, online allocation methods, and exact VCG.
- **Theoretical analysis** — provides analytical results, bounds, and mechanism properties.
- **Trace-driven evaluation** — evaluates workloads derived from the Microsoft Philly GPU cluster trace.
- **Slurm integration** — includes a functional auction broker and settlement workflow.

---

## RAVA Scoring Function

RAVA ranks submitted jobs using a reliability-aware allocation scoring function:

$$
S_i = \frac{
b_i \cdot g_{\kappa}(\rho_i) \cdot \eta_i^{\gamma}
\cdot (1+\lambda\phi_i)
}{D_i^{\alpha}}
$$

The scoring function considers the following factors:

- **Bid value (\(b_i\))** — Economic value submitted by the tenant.
- **Reliability adjustment (\(g_{\kappa}(\rho_i)\))** — Reliability-based weighting parameterized by the deposit rate \(\kappa\).
- **Tenant reliability (\(\rho_i\))** — Estimated probability of successful job execution.
- **Resource efficiency (\(\eta_i\))** — Resource-efficiency factor.
- **Efficiency exponent (\(\gamma\))** — Controls the influence of resource efficiency.
- **Flexibility factor (\(\phi_i\))** — Captures job scheduling flexibility.
- **Flexibility weight (\(\lambda\))** — Controls the contribution of flexibility.
- **Resource demand (\(D_i\))** — Demand-related allocation factor.
- **Demand exponent (\(\alpha\))** — Controls the demand penalty.

### Allocation Workflow

1. Calculate the allocation score for every submitted job.
2. Rank jobs in descending order of their scores.
3. Greedily allocate feasible jobs to available servers.
4. Support multi-server gang scheduling.
5. Determine critical bids for winning jobs.
6. Apply pay-on-use settlement and deposit rules.
7. Update tenant reliability between auction epochs.

## Installation

### Requirements

- Python ≥ 3.10
- pip
- Git

The implementation has been tested with Python 3.13.

### Clone the repository

```bash
git clone https://github.com/Sayan-2607/REVA_GPU.git
cd REVA_GPU
```

### Install dependencies

```bash
pip install -e ".[test]"
```

### Run the test suite

```bash
pytest -q
```

The repository includes 44 tests covering theory, mechanism properties, worked examples, and regression checks.

### Validate paper results

```bash
rava check
```

This checks reported paper values against the stored experiment summary in `results/summary.json`.

---

## Quick Start

Run a simple auction with tenant reputation and GPU cluster resources:

```python
from rava import RavaAuction, BetaReputation, jobs_from_records

# Initialize tenant reputation
rep = BetaReputation()
rep.counts.update({
    "alice": [18, 1],  # past runs / no-shows
    "bob": [2, 6]
})

# Define submitted jobs
jobs = [
    dict(
        user="alice",
        demand=[4, 16, 64],
        slots=3,
        window=(0, 5),
        value=60,
    ),
    dict(
        user="bob",
        demand=[8, 32, 128],
        slots=4,
        window=(0, 5),
        value=150,
    ),
]

# Define cluster capacity and generate jobs
J = jobs_from_records(
    [[8, 64, 256]],
    jobs,
    T=6,
    reputation=rep,
)

# Run the RAVA auction
result = RavaAuction(kappa=0.0).run(J)

print(result.winners)
print(result.start)
print(result.prices)
```

More runnable examples are available in `examples/`.

---

## Command-Line Interface

RAVA provides a CLI for running examples, experiments, analysis, theoretical evaluations, and Slurm operations.

| Command | Description | Paper Reference |
|---|---|---|
| `rava example` | Runs the six-job example across supported mechanisms | Section VI, Table III |
| `rava experiment main` | Main experiment | Section VIII-B |
| `rava experiment kappa` | Deposit parameter study | Section VIII-D |
| `rava experiment bad` | Reliability-related experiments | Section VIII-E |
| `rava experiment flex` | Flexibility experiment | Section VIII-F |
| `rava experiment regret` | Regret experiment | Section VIII-G |
| `rava experiment adversary` | Adversarial evaluation | Section VIII |
| `rava experiment overbid` | Overbidding experiment | Section VIII |
| `rava experiment scale` | Scaling evaluation | Section VIII |
| `rava philly --seeds 1,2` | Philly trace-driven experiments | Section VIII-C |
| `rava philly-overbook` | Trace-driven overbooking study | Section VIII-C |
| `rava philly-combo` | RAVA + overbooking evaluation | Section VIII-C |
| `rava tune-opd` | OPD parameter tuning | Section VIII-C |
| `rava analyze` | Recomputes summary statistics and figures | Section VIII |
| `rava check` | Validates reported paper numbers against stored results | All |
| `rava theory eps` | Theorem 2 evaluation | Section V |
| `rava theory exploit` | Proposition 4 evaluation | Section V |
| `rava theory mis` | Proposition 1 evaluation | Section V |
| `rava slurm auction BIDS STATE` | Runs the Slurm auction broker | Section VII |
| `rava slurm settle STATE` | Settles completed jobs | Section VII |

### Slurm Broker

```bash
rava slurm auction BIDS STATE --dry-run
rava slurm settle STATE
```

An optional `--cluster FILE` argument allows the broker to use an alternative cluster configuration.

---

## Repository Structure

```text
REVA_GPU/
├── src/
│   └── rava/
│       ├── model.py
│       ├── reputation.py
│       ├── mechanism.py
│       ├── placement.py
│       ├── auction.py
│       ├── settlement.py
│       ├── theory.py
│       ├── baselines.py
│       ├── bounds.py
│       ├── metrics.py
│       ├── stats.py
│       ├── workloads/
│       ├── experiments/
│       ├── analysis/
│       ├── integrations/
│       │   └── slurm_broker.py
│             └── cli.py
├── tests/
├── examples/
├── configs/
├── scripts/
├── data/
├── results/
├── slurm/
├── paper/
│   ├── main.tex
│   ├── main.pdf
│   ├── refs.bib
│   └── figures/
├── legacy_v1/
├── docs/
│   ├── ARCHITECTURE.md
│   └── REPRODUCE.md
├── CITATION.cff
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── LICENSE
```

### Core Modules

| Module | Responsibility |
|---|---|
| `model.py` | Job/server models, resource footprints, settlement utility, effective value |
| `reputation.py` | Beta reputation, forgetting, floors, newcomer prior |
| `mechanism.py` | RAVA scoring, greedy allocation, critical prices |
| `placement.py` | Numba-based placement engine, earliest-fit, best-fit, gang jobs |
| `auction.py` | Auction interface and result handling |
| `settlement.py` | Pay-on-use settlement, reserve adaptation, early-release credit |
| `theory.py` | Closed-form theoretical results |
| `baselines.py` | Alternative auction and allocation mechanisms |
| `bounds.py` | LP-based efficiency upper bound |
| `metrics.py`, `stats.py` | Evaluation metrics and statistics |
| `workloads/` | Synthetic workloads, worked example, Philly trace loader |
| `experiments/` | Experiments reported in Section VIII |
| `analysis/` | Figure generation and paper result aggregation |
| `integrations/` | Cluster integration, including the Slurm broker |
| `cli.py` | Command-line interface |

---

## Experiments & Evaluation

RAVA includes synthetic and trace-driven experiments to evaluate allocation performance, reliability-aware ranking, pricing, efficiency, scaling, and mechanism comparisons.

### Synthetic Evaluation

The synthetic experiments cover:

- Main mechanism evaluation
- Deposit parameter \(\kappa\) sensitivity
- Reliability and adverse behavior
- Workload flexibility
- Regret
- Adversarial scenarios
- Overbidding
- System scaling

### Trace-Driven Evaluation

The trace-driven experiments use the **Microsoft Philly GPU cluster trace**.

The derived dataset includes:

- 111,549 jobs
- 68,798 jobs with per-job GPU utilization data

Trace-derived fields include GPU demand, gang size, runtime, submission time, user identity, job outcome, tenant history, and measured GPU utilization.

Synthetic assumptions include job values, deadline slack, failed-job reservation lengths, and the 35% subsample.

---

## Reproducibility

### Analyze stored results

```bash
rava analyze
rava check
```

Recomputes statistics and figures from stored experiment outputs.

### Run the complete experiment suite

```bash
bash scripts/run_all.sh
```

The full reproduction pipeline is expected to take approximately one hour on a two-core machine.

### Rebuild Philly trace data

```bash
bash scripts/rebuild_philly_data.sh
```

This reconstructs the derived dataset from the public trace. It requires Git LFS and may take approximately 20 minutes.

For experimental seeds, train/test splits, configuration details, and the optional Slurm setup, see:

- `docs/REPRODUCE.md`
- `docs/ARCHITECTURE.md`
- `configs/paper_params.json`

---

## Slurm Integration

The repository includes a functional Slurm auction broker with:

- Bid submission
- Auction execution
- Cluster-state handling
- Allocation decisions
- Settlement
- Job-completion logging

The included Slurm configuration and logs demonstrate an end-to-end functional test on a single node using placeholder GPU devices.

**This is a functional integration test, not a production deployment or large-scale cluster evaluation.**

---

## Implementation Status & Scope

| Component | Status |
|---|---|
| Auction mechanism | Implemented |
| Critical-bid pricing | Implemented |
| Beta reputation model | Implemented |
| Pay-on-use settlement | Implemented |
| Multi-resource placement | Implemented |
| Multi-server gang jobs | Implemented |
| Synthetic experiments | Implemented |
| Philly trace evaluation | Implemented |
| Theoretical analysis | Implemented |
| Baseline mechanisms | Implemented |
| Slurm broker | Functional prototype |
| Production cluster deployment | Not evaluated |
| Kubernetes integration | Future work |

---

## Limitations & Assumptions

- Trace-driven evaluations use real workload characteristics but include synthetic economic attributes.
- The Philly dataset is derived and processed rather than used as a direct unmodified workload replay.
- The Slurm integration is a prototype validated using placeholder GPU devices.
- Large-scale production deployment has not been evaluated.
- Kubernetes integration is outside the current scope.

These limitations distinguish implemented research functionality from production infrastructure validation.

---

## License & Data Attribution

### Code

Licensed under the **MIT License**. See [`LICENSE`](LICENSE).

### Dataset

The derived Philly data is based on Microsoft Philly cluster traces and is distributed under **CC BY 4.0**, attributed to Jeon et al., USENIX ATC 2019.

---

## Citation

If you use RAVA in your research, please cite the associated paper:

```bibtex
@article{rava,
  title   = {RAVA: A Reciprocal Adaptive Value and Resource Allocation Auction for Reliability-Aware GPU and Cloud Clusters},
  author  = {Sayan Ghosh},
  journal = {Research manuscript},
  year   = {2026}
}
```

For formal citation metadata, refer to `CITATION.cff`.

---

## Acknowledgements

RAVA builds on research in resource allocation, auction mechanisms, cluster scheduling, reliability modelling, and GPU workload analysis.

---

**RAVA — Reliability-aware allocation for shared computing resources.**
