#!/usr/bin/env bash
# Rebuild data/ from the public Microsoft Philly trace (CC BY 4.0). Needs git-lfs, ~8 GB RAM, ~20 min.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p build && cd build
[ -d philly-traces ] || git clone https://github.com/msr-fiddle/philly-traces
cd philly-traces && git lfs pull
echo "2037ccf63a725a02be718b0e1aeee6142c90b7f62d0d849295120e9f6ba13d2c  trace-data.tar.gz" | sha256sum -c -
tar -xzf trace-data.tar.gz
cd ../..
rava prepare-philly build/philly-traces/trace-data/cluster_job_log data/philly_jobs.csv.gz
( cd data && rava philly-util ../build/philly-traces/trace-data && gzip -f philly_job_util.csv )
