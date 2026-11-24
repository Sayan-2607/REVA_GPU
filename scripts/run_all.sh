#!/usr/bin/env bash
# Re-run every experiment of the paper from scratch, then rebuild summary.json, all figures and the PDF.
# About 1 hour on 2 cores. Results go to results/ (existing files are overwritten).
set -euo pipefail
cd "$(dirname "$0")/.."
for e in main kappa bad flex regret adversary overbid scale; do rava experiment "$e"; done
rava philly --seeds 1,2
rava philly --seeds 3
rava philly --seeds 1 --killed-ok
rava philly-overbook
rava philly-combo
rava tune-opd
# Exact VCG at n=40 (10 workloads, ~20 min) and the legacy figures of Sections VI and VIII-G:
( cd legacy_v1 && python3 run_vcg.py && cp results_vcg.json ../results/res_vcg_v1.json )
rava analyze
rava check
( cd paper && latexmk -pdf -interaction=nonstopmode main.tex )
