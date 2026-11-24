#!/usr/bin/env python3
"""
RAVA broker for Slurm.

  rava_slurm.py auction  <bids.json> <state_dir> [--slot SEC] [--kappa K]
        reads node capacity from `scontrol show node -o`, runs RAVA (core.py), submits every winner with
        `sbatch --begin=<slot start> --time=<slots> --gres=gpu:<g> --cpus-per-task --mem --nodelist`,
        and writes the epoch ledger (prices, planned starts, Slurm job ids).
  rava_slurm.py settle   <state_dir> [--kappa K] [--zeta Z]
        reads Slurm's job-completion log (jobcomp/filetxt), charges each winner its price if the job
        COMPLETED (pay-on-use) or forfeits kappa*price otherwise, and updates the Beta reputation store.

bids.json: [{"name","user","gpus","cpus","mem_gb","slots","window":[a,e],"bid","command"}, ...]
"""
import sys, os, json, time, re, subprocess, argparse, datetime as dt, numpy as np
from rava import core as C

A0, B0 = 4.0, 1.0

def sh(cmd):
    return subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True).stdout

def cluster(path=None):
    """Node capacities from `scontrol show node -o`, or from a JSON file [{name,gpu,cpu,mem}] in dry-run mode."""
    if path:
        return json.load(open(path))
    nodes = []
    for line in sh("scontrol show node -o").strip().splitlines():
        f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        gpus = int(re.search(r"gpu(?::[^:,]+)?:(\d+)", f.get("Gres", "gpu:0")).group(1)) if "gpu" in f.get("Gres", "") else 0
        nodes.append(dict(name=f["NodeName"], gpu=gpus, cpu=int(f["CPUTot"]), mem=int(f["RealMemory"]) / 1024))
    return nodes

def load_rep(state):
    p = os.path.join(state, "reputation.json")
    return json.load(open(p)) if os.path.exists(p) else {}

def auction(bids_path, state, slot, kappa, cluster_file=None, dry_run=False):
    os.makedirs(state, exist_ok=True)
    bids = json.load(open(bids_path)); nodes = cluster(cluster_file); rep = load_rep(state)
    cap = np.array([[n["gpu"], n["cpu"], n["mem"]] for n in nodes], float)
    T = max(b["window"][1] for b in bids) + 1
    D = np.array([[b["gpus"], b["cpus"], b["mem_gb"]] for b in bids], float)
    L = np.array([b["slots"] for b in bids]); Aw = np.array([b["window"][0] for b in bids]); Ew = np.array([b["window"][1] for b in bids])
    rho = np.array([(A0 + rep.get(b["user"], [0, 0])[0]) / (A0 + B0 + sum(rep.get(b["user"], [0, 0]))) for b in bids])
    bid = np.array([b["bid"] for b in bids], float)
    J = C.Jobs(cap, T, D, np.ones(len(bids), int), L, Aw, Ew, bid, rho, rho, np.ones(len(bids)), np.ones(len(bids)), np.zeros(len(bids), int))
    t0 = time.time(); st, nd, p = C.greedy(J, bid, C.rava_weight(J, kappa)); t_auction = time.time() - t0
    epoch_start = dt.datetime.now().replace(microsecond=0) + dt.timedelta(seconds=15)
    ledger = dict(epoch_start=epoch_start.isoformat(), slot=slot, kappa=kappa, auction_seconds=t_auction, jobs=[])
    for i, b in enumerate(bids):
        row = dict(name=b["name"], user=b["user"], bid=b["bid"], rho=float(rho[i]), score=float(bid[i] * C.rava_weight(J, kappa)[i]),
                   won=bool(st[i] >= 0), price=float(p[i]), planned_slot=int(st[i]))
        if st[i] >= 0:
            begin = epoch_start + dt.timedelta(seconds=int(st[i]) * slot)
            minutes = max(1, int(np.ceil(b["slots"] * slot / 60)))
            cmd = (f"sbatch --parsable -J rava_{b['name']} --begin={begin.strftime('%Y-%m-%dT%H:%M:%S')} --time={minutes} "
                   f"{("--gres=gpu:" + str(b["gpus"])) if b["gpus"] > 0 else ""} --cpus-per-task={b['cpus']} --mem={b['mem_gb']}G --nodelist={nodes[nd[i, 0]]['name']} "
                   f"-o {state}/{b['name']}.out --wrap=\"{b['command']}\"")
            row["sbatch"] = cmd; row["begin"] = begin.isoformat()
            row["slurm_id"] = "dry-run" if dry_run else sh(cmd).strip()
        ledger["jobs"].append(row)
    json.dump(ledger, open(os.path.join(state, "ledger.json"), "w"), indent=1)
    for r in ledger["jobs"]:
        print(f"{r['name']:4s} user={r['user']:10s} bid={r['bid']:6.1f} rho={r['rho']:.2f} score={r['score']:6.2f} "
              f"{'WIN  slot ' + str(r['planned_slot']) + ' price ' + format(r['price'], '.2f') + ' job ' + r['slurm_id'] if r['won'] else 'lose'}")
    print(f"auction time {t_auction*1000:.1f} ms")

def settle(state, kappa, zeta, jobcomp="/var/log/slurm/jobcomp.log"):
    """Charge on use, update reputation from Slurm's job-completion log."""
    ledger = json.load(open(os.path.join(state, "ledger.json"))); rep = load_rep(state)
    comp = {}
    for line in open(jobcomp):
        f = dict(kv.split("=", 1) for kv in line.split() if "=" in kv)
        comp[f["JobId"]] = f
    bill = []
    for r in ledger["jobs"]:
        if not r["won"]: continue
        f = comp.get(r["slurm_id"])
        if f is None: print("not finished yet:", r["name"]); continue
        ok = f["JobState"] == "COMPLETED"
        s, fl = rep.get(r["user"], [0.0, 0.0]); s *= zeta; fl *= zeta
        rep[r["user"]] = [s + ok, fl + (not ok)]
        charge = r["price"] if ok else kappa * r["price"]
        planned = dt.datetime.fromisoformat(r["begin"]); actual = dt.datetime.fromisoformat(f["StartTime"])
        bill.append(dict(name=r["name"], user=r["user"], state=f["JobState"], charge=round(charge, 2),
                         start_delay_s=(actual - planned).total_seconds(), new_rho=round((A0 + rep[r["user"]][0]) / (A0 + B0 + sum(rep[r["user"]])), 3)))
    json.dump(rep, open(os.path.join(state, "reputation.json"), "w"), indent=1)
    json.dump(bill, open(os.path.join(state, "settlement.json"), "w"), indent=1)
    for b in bill: print(b)

def main(argv=None):
    import sys as _s
    if argv is not None: _s.argv = ["rava-slurm"] + list(argv)
    _main()


def _main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("a"); ap.add_argument("b", nargs="?")
    ap.add_argument("--slot", type=int, default=3600); ap.add_argument("--kappa", type=float, default=0.0); ap.add_argument("--zeta", type=float, default=1.0)
    ap.add_argument("--cluster", default=None, help="JSON node list (skip scontrol)"); ap.add_argument("--dry-run", action="store_true", help="plan only; do not call sbatch")
    ap.add_argument("--jobcomp", default="/var/log/slurm/jobcomp.log", help="Slurm jobcomp/filetxt log")
    x = ap.parse_args()
    if x.cmd == "auction": auction(x.a, x.b, x.slot, x.kappa, x.cluster, x.dry_run)
    elif x.cmd == "settle": settle(x.a, x.kappa, x.zeta, x.jobcomp)


if __name__ == "__main__":
    _main()
