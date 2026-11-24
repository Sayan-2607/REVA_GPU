"""Beta reputation with forgetting, floor and newcomer prior (Equation (5), Propositions 4 and 5)."""
import json
from dataclasses import dataclass, field


@dataclass
class BetaReputation:
    a0: float = 4.0          # prior pseudo-successes
    b0: float = 1.0          # prior pseudo-failures
    zeta: float = 1.0        # forgetting factor (1 = no forgetting)
    rho_min: float = 0.05    # reputation floor (keeps g_kappa bounded away from 0, Lemma 3)
    counts: dict = field(default_factory=dict)   # user -> [s, f]

    @property
    def newcomer(self):
        """rho_0 = max(rho_min, a0 / (a0 + b0))."""
        return max(self.rho_min, self.a0 / (self.a0 + self.b0))

    def rho(self, user):
        s, f = self.counts.get(user, (0.0, 0.0))
        return max(self.rho_min, (self.a0 + s) / (self.a0 + self.b0 + s + f))

    def mass(self, user):
        """N = a0 + b0 + s + f (used by the epsilon bound of Theorem 2)."""
        s, f = self.counts.get(user, (0.0, 0.0))
        return self.a0 + self.b0 + s + f

    def update(self, user, ran):
        """After an epoch in which `user` won: s <- zeta*s + 1[ran], f <- zeta*f + 1[no-show]."""
        s, f = self.counts.get(user, (0.0, 0.0))
        self.counts[user] = [self.zeta * s + (1.0 if ran else 0.0), self.zeta * f + (0.0 if ran else 1.0)]
        return self.rho(user)

    def whitewash_profitable(self, user):
        """Proposition 5: a fresh identity raises the score iff rho_0 > rho_i (for kappa < 1)."""
        return self.newcomer > self.rho(user)

    def save(self, path):
        json.dump(dict(a0=self.a0, b0=self.b0, zeta=self.zeta, rho_min=self.rho_min, counts=self.counts), open(path, "w"), indent=1)

    @classmethod
    def load(cls, path):
        d = json.load(open(path)); return cls(**d)
