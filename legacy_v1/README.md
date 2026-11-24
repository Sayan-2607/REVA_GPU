# legacy_v1 — first-version simulator

Used for three things in the final paper, on exactly the same synthetic workloads as `rava` (validated: identical welfare
and payments):

* exact VCG at n = 40 (`run_vcg.py` → `results_vcg.json`, copied to `../results/res_vcg_v1.json`);
* the worked-example, ranking, GPU-day and α–γ heat-map figures (`example.py`, `extra_data.py`, `make_figs.py`,
  `make_figs2.py` → `legacy_figures/`, then copied to `../paper/figures/`);
* the original 20-epoch reputation study (`rava_sim.py epoch`).

`results_*.json` are the raw v1 results these scripts read.
