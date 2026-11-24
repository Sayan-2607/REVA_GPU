"""Per-job mean GPU utilisation from the Philly trace (10-minute samples)."""
import json, sys, numpy as np, pandas as pd
D = sys.argv[1]
cols = ["time", "machineId"] + [f"gpu{k}_util" for k in range(8)]
parts = []
for ch in pd.read_csv(f"{D}/cluster_gpu_util", usecols=range(10), names=cols, header=0, chunksize=2_000_000, dtype=str):
    ch = ch[ch["time"].str[15] == "0"]                     # keep minutes ending in 0
    t = pd.to_datetime(ch["time"].str[:19], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    u = ch[cols[2:]].apply(pd.to_numeric, errors="coerce").astype("float32")
    u.insert(0, "machine", ch["machineId"].values); u.insert(0, "ts", t.values.astype("datetime64[s]").astype(np.int64))
    parts.append(u.dropna(subset=["ts"])); print("chunk", len(parts), flush=True)
U = pd.concat(parts); del parts
U.sort_values(["machine", "ts"], inplace=True)
by = {m: g for m, g in U.groupby("machine")}
J = json.load(open(f"{D}/cluster_job_log"))
rows = []
for j in J:
    if not j["attempts"]: continue
    a = j["attempts"][-1]
    if not a.get("start_time") or not a.get("end_time") or a["start_time"] == "None" or a["end_time"] == "None": continue
    s = pd.Timestamp(a["start_time"]).value // 10**9; e = pd.Timestamp(a["end_time"]).value // 10**9
    vals = []
    for d in a["detail"]:
        g = by.get(d["ip"])
        if g is None: continue
        ts = g["ts"].values; lo, hi = np.searchsorted(ts, s), np.searchsorted(ts, e)
        if hi <= lo: continue
        for gp in d["gpus"]:
            c = f"{gp}_util"
            if c in g: vals.append(np.nanmean(g[c].values[lo:hi]))
    if vals: rows.append((j["jobid"], j["user"], float(np.nanmean(vals)), len(vals)))
pd.DataFrame(rows, columns=["jobid", "user", "gpu_util", "n_gpus_measured"]).to_csv("philly_job_util.csv", index=False)
print("jobs with util:", len(rows))
