"""Flat namespace used by the experiment scripts (kept for exact reproducibility of the paper's runs)."""
from .placement import _cands, try_place, init_R, place, critical_other
from .model import Jobs, effective_value, expected_utility, footprint
from .mechanism import greedy, rava_weight, g_kappa
from .baselines import opd, _opd, drf, overbook, mcafee, first_price, second_price, los, gvmpac2, vcg
from .bounds import lp_bound
from .metrics import metrics
from .workloads.synthetic import synthetic
