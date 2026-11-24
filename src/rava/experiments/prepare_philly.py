"""Build data/philly_jobs.csv.gz from the public Philly job log (trace-data/cluster_job_log)."""
import sys, json, pandas as pd
J = json.load(open(sys.argv[1])); rows = []
for j in J:
    a = j["attempts"]
    if not a or not a[0]["detail"]: continue
    g = sum(len(d["gpus"]) for d in a[0]["detail"])
    try: st = pd.Timestamp(a[0]["start_time"]); en = pd.Timestamp(a[-1]["end_time"])
    except Exception: continue
    if pd.isna(st) or pd.isna(en): continue
    rows.append((j["jobid"], j["user"], j["vc"], j["status"], g, pd.Timestamp(j["submitted_time"]), (en - st).total_seconds() / 3600, len(a)))
pd.DataFrame(rows, columns=["jobid", "user", "vc", "status", "gpus", "submit", "hours", "attempts"]).to_csv(sys.argv[2], index=False)
print(len(rows), "jobs")
