"""Evaluate the closed-form results of Section V."""
from rava import theory as th
print("Prop. 1  overconfident tenant (q=0.5, qhat=1): mu at kappa=0, 0.5, 1 ->",
      [round(th.misspecification(0.5, 1.0, k), 3) for k in (0, 0.5, 1)])
for N in (5, 20, 100):
    print(f"Thm 2   epsilon bound, history mass N={N:3d}: {th.epsilon_bound(0.9, 0.0, 100, 0.05, N):8.2f}")
print("Thm 2   with forgetting zeta=0.8:", round(th.epsilon_bound(0.9, 0.0, 100, 0.05, 0, zeta=0.8), 2))
print("Prop. 4  defaults before rho<0.6: zeta=1, 100 past successes ->", th.defaults_before_threshold_no_forgetting(100, 0, 0.6),
      "| zeta=0.8, any history ->", th.max_defaults_with_forgetting(0.6, zeta=0.8))
