"""The worked example of Section VI: one 8-GPU / 64-core / 256-GB server, four slots, six tenants."""
import numpy as np
from ..model import jobs_from_records

CAPACITY = [[8, 64, 256]]
T = 4
RECORDS = [
    dict(name="J1", task="LLM fine-tune",         user="llm-team",  demand=[6, 24, 128], slots=3, window=(0, 3), value=130, rho=0.55),
    dict(name="J2", task="Vision training",       user="vision",    demand=[4, 16, 64],  slots=2, window=(0, 3), value=48,  rho=0.95),
    dict(name="J3", task="Inference burst",       user="inference", demand=[2, 8, 32],   slots=2, window=(1, 2), value=30,  rho=0.98),
    dict(name="J4", task="Hyper-parameter sweep", user="hpo",       demand=[4, 16, 64],  slots=2, window=(0, 3), value=40,  rho=0.90),
    dict(name="J5", task="CPU preprocessing",     user="etl",       demand=[0, 32, 128], slots=2, window=(0, 3), value=18,  rho=0.97),
    dict(name="J6", task="Student RL project",    user="student",   demand=[2, 8, 32],   slots=4, window=(0, 3), value=22,  rho=0.92),
]


def worked_example(j1_bid=130):
    recs = [dict(r) for r in RECORDS]
    recs[0]["value"] = j1_bid
    J = jobs_from_records(CAPACITY, recs, T)
    J.names = [r["name"] for r in recs]
    return J
