"""RAVA: Reciprocal Adaptive Value and Resource Allocation Auction.

Reference implementation of the paper "RAVA: A Reciprocal Adaptive Value and Resource Allocation Auction for
Reliability-Aware GPU and Cloud Clusters".
"""
__version__ = "1.0.0"
from .model import Jobs, jobs_from_records, effective_value, expected_utility, footprint
from .mechanism import greedy, rava_weight, g_kappa
from .auction import RavaAuction, AuctionResult
from .reputation import BetaReputation
from .settlement import charge, expected_revenue, update_reserve, early_release_credit
from .baselines import first_price, second_price, los, gvmpac2, opd, drf, overbook, mcafee, vcg
from .bounds import lp_bound
from .metrics import metrics
from .workloads.synthetic import synthetic
from . import theory, stats

__all__ = ["Jobs", "jobs_from_records", "effective_value", "expected_utility", "footprint", "greedy", "rava_weight",
           "g_kappa", "RavaAuction", "AuctionResult", "BetaReputation", "charge", "expected_revenue", "update_reserve",
           "early_release_credit", "first_price", "second_price", "los", "gvmpac2", "opd", "drf", "overbook", "mcafee",
           "vcg", "lp_bound", "metrics", "synthetic", "theory", "stats"]
