import json, os, subprocess, sys, shutil
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def run(*args):
    return subprocess.run([sys.executable, "-m", "rava.cli", *args], capture_output=True, text=True, cwd=ROOT,
                          env=dict(os.environ, PYTHONPATH=os.path.join(ROOT, "src")))


def test_example_cli():
    out = run("example").stdout
    assert "J2,J3,J4,J5,J6" in out and "148.7" in out and "46.02" in out


def test_theory_cli():
    assert "epsilon bound" in run("theory", "eps", "--mass", "25").stdout


def test_slurm_dry_run(tmp_path):
    shutil.copy(os.path.join(ROOT, "slurm", "bids.json"), tmp_path / "bids.json")
    json.dump({"llm-team": [7, 8], "vision": [15, 0], "inference": [45, 0], "hpo": [14, 1], "etl": [28, 0], "student": [19, 1]},
              open(tmp_path / "reputation.json", "w"))
    r = run("slurm", "auction", str(tmp_path / "bids.json"), str(tmp_path), "--slot", "20", "--kappa", "0",
            "--cluster", os.path.join(ROOT, "slurm", "cluster_example.json"), "--dry-run")
    led = json.load(open(tmp_path / "ledger.json"))
    won = {j["name"]: round(j["price"], 2) for j in led["jobs"] if j["won"]}
    assert won == {"J2": 46.02, "J3": 28.57, "J4": 0.0, "J5": 0.0, "J6": 0.0}
