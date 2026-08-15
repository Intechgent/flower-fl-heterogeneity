# Raw experimental results

Setup: CIFAR-10, small CNN (2 conv + 3 fc), FedAvg, 10 simulated clients, 30 rounds,
1 local epoch per round, batch size 32, SGD with momentum 0.9, seed 1.
Partitioning: DirichletPartitioner by label.
Metric: centralised accuracy (%) on the full CIFAR-10 test set, evaluated server-side each round.
Rounds listed 0 to 30, where round 0 is the untrained global model.

All runs pass the client count explicitly:

    flwr run . --stream --federation-config="num-supernodes=10"

## Summary

| alpha | lr=0.01 | lr=0.1 |
|---|---|---|
| 10 (mild skew) | 60.0 | 29.2 |
| 0.5 (moderate) | 56.6 | 29.1 |
| 0.1 (severe) | 49.4 | 18.0 |

## lr = 0.01

alpha=10
9.4, 17.7, 29.7, 35.3, 40.1, 42.7, 44.9, 46.4, 48.1, 49.6, 50.6, 51.6, 52.3, 53.5, 53.9, 54.9, 55.4, 56.0, 56.5, 56.8, 57.4, 57.7, 58.0, 58.3, 58.7, 58.8, 59.1, 59.6, 60.0, 60.0, 60.0

alpha=0.5
9.4, 11.0, 24.7, 31.7, 35.4, 38.4, 40.5, 42.8, 44.4, 45.6, 46.8, 47.8, 48.8, 49.5, 50.5, 51.4, 52.6, 53.0, 53.7, 54.1, 54.3, 54.6, 55.0, 55.1, 55.5, 55.5, 55.6, 56.0, 56.4, 56.4, 56.6

alpha=0.1
9.4, 10.0, 18.6, 24.9, 27.6, 29.0, 30.5, 33.2, 35.7, 38.3, 40.8, 42.3, 43.5, 44.7, 45.6, 46.0, 47.1, 47.6, 47.9, 48.2, 48.6, 48.6, 49.1, 49.4, 49.6, 49.2, 49.3, 49.5, 49.1, 48.8, 49.4

## lr = 0.1

alpha=10
9.4, 10.0, 13.8, 21.0, 20.6, 16.3, 18.6, 18.3, 18.2, 20.6, 22.5, 21.2, 21.7, 23.0, 23.5, 24.6, 24.4, 24.0, 24.6, 25.8, 26.1, 23.7, 27.5, 27.3, 26.2, 28.5, 30.3, 28.9, 29.6, 24.9, 29.2

alpha=0.5
9.4, 10.0, 10.0, 16.8, 17.6, 19.4, 20.9, 23.0, 23.8, 24.3, 25.4, 25.4, 27.3, 25.9, 27.1, 28.9, 26.4, 27.5, 28.5, 29.1, 27.4, 26.7, 28.7, 28.3, 27.8, 28.1, 28.6, 29.0, 29.0, 29.0, 29.1

alpha=0.1
9.4, 10.0, 10.0, 10.6, 16.1, 13.5, 14.9, 17.2, 17.9, 17.4, 16.5, 18.3, 17.4, 16.9, 16.4, 16.4, 16.6, 18.3, 16.8, 18.7, 17.4, 19.9, 17.9, 20.0, 17.9, 18.1, 17.5, 19.8, 16.8, 17.3, 18.0

## Observations

At lr=0.01, accuracy decreases monotonically with heterogeneity: 60.0 / 56.6 / 49.4, a spread of
10.6 points. All three curves are smooth, with no oscillation or divergence. Convergence speed also
degrades: alpha=10 passes 50% by round 10, alpha=0.5 by round 15, alpha=0.1 never does.

At lr=0.1 every run is far worse, and alpha=10 and alpha=0.5 become indistinguishable (29.2 vs 29.1).
Severe heterogeneity still shows through at 18.0. So a poorly tuned learning rate compresses the
mild-to-moderate distinction into nothing while severe skew remains visible.

Client drift is visible in the train/test gap: at lr=0.1, alpha=0.1 has the lowest training loss of
any run (0.92) and the worst global accuracy (18.0). Clients fit their narrow local label
distributions well while the averaged global model does not.

## Notes on discarded runs

An earlier set of runs was discarded: Flower's stored simulation config silently reverted from 10
SuperNodes to the default 2 partway through the session, so client count was not controlled across
conditions. Those runs also showed a complete divergence to random-guessing accuracy at alpha=10
under lr=0.1, and between-seed differences at fixed alpha larger than the differences between alpha
values.

## Environment note

charset-normalizer is pinned below 3.5.1 in pyproject.toml. Version 3.5.1 was published mid-session
with no Windows wheel, which broke Flower's per-run dependency install (exit code 608).

## Notes on discarded runs

An earlier set of runs was discarded: Flower's stored simulation config silently reverted from 10
SuperNodes to the default 2 partway through the session, so client count was not controlled.

Those runs showed severe instability at lr=0.1, including a complete divergence to random-guessing
accuracy at alpha=10 and an 8-point disagreement between seeds at alpha=0.5.

The controlled lr=0.1 runs at 10 clients did not show this: all three alpha values climbed steadily
with no divergence, suggesting client count rather than learning rate was the main driver. With only
2 clients each client's drift is half the global update, leaving little averaging to absorb it. This
is not yet confirmed, as the earlier runs also varied by seed; a controlled run at 2 clients with
lr=0.1 would settle it.