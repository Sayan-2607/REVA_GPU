"""High-level API.

>>> from rava import RavaAuction, jobs_from_records
>>> J = jobs_from_records([[8, 64, 256]], records, T=4)
>>> res = RavaAuction(kappa=0.0).run(J)
>>> res.winners, res.prices
"""
from dataclasses import dataclass
import numpy as np
from .mechanism import greedy, rava_weight
from .metrics import metrics


@dataclass
class AuctionResult:
    start: np.ndarray        # start slot per job (-1 = lost)
    servers: np.ndarray      # server indices per job (-1 padded)
    prices: np.ndarray       # critical price per job (0 for losers)
    bids: np.ndarray
    weights: np.ndarray
    scores: np.ndarray

    @property
    def winners(self):
        return np.where(self.start >= 0)[0]


class RavaAuction:
    """RAVA with the paper's defaults: kappa=0, alpha=0.5, gamma=0, lambda=0.2, no reserve."""

    def __init__(self, kappa=0.0, alpha=0.5, gamma=0.0, lam=0.2, reserve=0.0, use_reliability=True):
        self.kappa, self.alpha, self.gamma, self.lam = kappa, alpha, gamma, lam
        self.reserve, self.use_reliability = reserve, use_reliability

    def weights(self, J, pi=None):
        return rava_weight(J, self.kappa, alpha=self.alpha, gamma=self.gamma, lam=self.lam, pi=pi, use_rho=self.use_reliability)

    def run(self, J, bids=None, payments=True, pi=None):
        """Run Algorithms 1 and 2. If `bids` is None, tenants bid their effective value (Lemma 1)."""
        b = J.bids(self.kappa) if bids is None else np.asarray(bids, float)
        w = self.weights(J, pi)
        start, servers, p = greedy(J, b, w, pay="critical" if payments else "none", reserve=self.reserve)
        return AuctionResult(start, servers, p, b, w, b * w)

    def evaluate(self, J, result):
        return metrics(J, result.start, result.prices, self.kappa)
