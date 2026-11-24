# Reproducing the paper

1. Install: `pip install -e ".[test]"` (Python ≥ 3.10; tested with Python 3.13, versions in `requirements.txt`).
2. Check the installation: `pytest -q` (44 tests, ~15 s) and `rava check` (compares `results/summary.json` with the
   numbers stated in the paper).
3. Regenerate every number and figure from the stored raw results: `rava analyze` (~1 min).
4. Re-run all experiments from scratch: `bash scripts/run_all.sh` (~1 h on 2 cores, plus ~20 min for exact VCG).
5. Rebuild `data/` from the public Philly trace: `bash scripts/rebuild_philly_data.sh`.

Seeds: synthetic workload `s` at size `n` uses seed `1000*n + s`; the scalability runs use `900000 + n + s`;
Philly epochs use seeds 1–3 (validation: 2017-10-03..16, test: 2017-10-17..11-30).

Live Slurm (optional): install Slurm ≥ 23.11, use `slurm/slurm.conf` and `slurm/gres.conf` (placeholder GPU files
`/dev/fakegpu[0-7]` if the host has no GPUs), then
`rava slurm auction slurm/bids.json STATE --slot 20` and, after the jobs end, `rava slurm settle STATE`.
Without Slurm, add `--cluster slurm/cluster_example.json --dry-run`.
