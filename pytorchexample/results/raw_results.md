# Raw experimental results

Setup: CIFAR-10, small CNN, FedAvg, 10 clients, 30 rounds, local-epochs=1, lr=0.1, SGD momentum=0.9.
Metric: server-side (centralised) accuracy on the full CIFAR-10 test set, per round.

## Seed 42

alpha=10  (diverged at round 5, stuck at ~10% until round 20)
10.2, 16.6, 22.8, 25.1, 17.0, 10.0, 10.0, 10.1, 10.0, 10.1, 10.0, 10.0, 10.4, 10.3, 10.2, 10.6, 11.1, 10.9, 11.5, 10.2, 12.6, 15.9, 15.9, 12.2, 12.9, 13.3, 16.2, 16.7, 13.2, 16.4, 16.7

alpha=0.5
10.2, 15.0, 17.3, 21.3, 20.1, 20.2, 22.3, 23.2, 23.3, 21.6, 23.9, 23.7, 23.7, 21.2, 23.0, 23.9, 23.7, 24.4, 23.5, 23.1, 23.5, 23.6, 24.0, 24.0, 17.2, 21.5, 22.3, 23.5, 23.1, 22.4, 24.1

alpha=0.1
10.2, 18.2, 16.4, 15.9, 17.8, 16.9, 17.1, 17.2, 17.2, 17.6, 16.6, 18.4, 19.0, 21.2, 17.7, 20.2, 17.1, 19.6, 18.4, 20.0, 18.0, 19.7, 17.4, 19.5, 19.4, 20.8, 21.6, 11.6, 10.4, 17.2, 18.3

## Seed 1

alpha=10
9.4, 13.2, 17.6, 20.2, 20.8, 23.4, 22.3, 21.6, 23.9, 22.5, 20.2, 23.5, 22.8, 23.2, 23.4, 24.2, 24.7, 24.2, 24.2, 24.3, 24.1, 24.4, 24.3, 24.9, 23.8, 21.7, 24.8, 23.3, 24.8, 25.0, 22.7

alpha=0.5
9.4, 16.8, 18.3, 15.2, 17.0, 17.2, 16.6, 13.2, 19.2, 17.3, 19.0, 14.9, 17.2, 16.7, 14.4, 16.6, 17.4, 17.3, 16.9, 15.6, 11.1, 16.3, 16.3, 15.7, 15.5, 16.8, 16.2, 10.7, 15.9, 14.9, 15.9

alpha=0.1
9.4, 13.5, 16.4, 17.1, 16.7, 17.3, 17.0, 16.6, 17.8, 17.7, 17.2, 19.4, 17.0, 17.4, 18.0, 17.4, 16.5, 17.7, 18.1, 18.3, 16.0, 19.7, 18.3, 16.6, 20.4, 17.6, 19.7, 13.6, 16.3, 16.0, 18.1

## Summary

| alpha | seed 42 final | seed 1 final | seed 42 best | seed 1 best |
|---|---|---|---|---|
| 10 | 16.7 (diverged) | 22.7 | 25.1 | 25.0 |
| 0.5 | 24.1 | 15.9 | 24.4 | 19.2 |
| 0.1 | 18.3 | 18.1 | 21.6 | 20.4 |

## Observations

- alpha=0.1 replicates consistently across seeds (18.3 / 18.1 final), with characteristic
  oscillation and no stable convergence.
- alpha=10 and alpha=0.5 are highly seed-dependent. At alpha=0.5 the two seeds differ by 8 points:
  seed 42 converged smoothly to ~24%, seed 1 never converged.
- One alpha=10 run (seed 42) diverged entirely at round 5, sitting at ~10% (random guessing,
  train loss pinned at ln(10)=2.303) for 15 rounds before partially recovering.
- Between-seed variance at fixed alpha exceeds the between-alpha differences, so with only two
  seeds no clean alpha-to-performance relationship can be claimed.
- Hypothesis: lr=0.1 with SGD momentum=0.9 is unstable for this setup, amplifying seed differences
  and swamping the heterogeneity effect. Next step: rerun at a lower learning rate.