# Raw experimental results

Setup: CIFAR-10, small CNN (2 conv + 3 fc), FedAvg, 30 rounds, batch size 32, SGD with momentum 0.9,
seed 1. Client count, learning rate, and local epochs vary by experiment and are stated per section.
Partitioning: DirichletPartitioner by label.
Metric: centralised accuracy (%) on the full CIFAR-10 test set, evaluated server-side each round.
Rounds listed 0 to 30, where round 0 is the untrained global model.

All runs pass the client count explicitly on the command line, for example:

    flwr run . --stream --federation-config="num-supernodes=10"

Runs are fully reproducible: repeating a configuration with the same seed produced bit-identical
results on every occasion this was tested.

## Summary: alpha vs learning rate (1 local epoch, 10 clients)

| alpha | lr=0.01 | lr=0.1 |
|---|---|---|
| 10 (mild skew) | 60.0 | 29.2 |
| 0.5 (moderate) | 56.6 | 29.1 |
| 0.1 (severe) | 49.4 | 18.0 |

## lr = 0.01 (1 local epoch, 10 clients)

alpha=10
9.4, 17.7, 29.7, 35.3, 40.1, 42.7, 44.9, 46.4, 48.1, 49.6, 50.6, 51.6, 52.3, 53.5, 53.9, 54.9, 55.4, 56.0, 56.5, 56.8, 57.4, 57.7, 58.0, 58.3, 58.7, 58.8, 59.1, 59.6, 60.0, 60.0, 60.0

alpha=0.5
9.4, 11.0, 24.7, 31.7, 35.4, 38.4, 40.5, 42.8, 44.4, 45.6, 46.8, 47.8, 48.8, 49.5, 50.5, 51.4, 52.6, 53.0, 53.7, 54.1, 54.3, 54.6, 55.0, 55.1, 55.5, 55.5, 55.6, 56.0, 56.4, 56.4, 56.6

alpha=0.1
9.4, 10.0, 18.6, 24.9, 27.6, 29.0, 30.5, 33.2, 35.7, 38.3, 40.8, 42.3, 43.5, 44.7, 45.6, 46.0, 47.1, 47.6, 47.9, 48.2, 48.6, 48.6, 49.1, 49.4, 49.6, 49.2, 49.3, 49.5, 49.1, 48.8, 49.4

## lr = 0.1 (1 local epoch, 10 clients)

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

## Local epochs (lr=0.01, seed 1, 10 clients, 30 rounds)

| local epochs | alpha=10 | alpha=0.5 | alpha=0.1 |
|---|---|---|---|
| 1 | 60.0 | 56.6 | 49.4 |
| 5 | 55.7 | 51.7 | 42.8 |
| difference | -4.3 | -4.9 | -6.6 |

alpha=10, 5 local epochs
9.4, 39.5, 49.6, 54.1, 55.6, 55.9, 56.6, 57.1, 56.8, 56.9, 56.4, 56.8, 56.9, 56.8, 57.0, 56.8, 56.8, 56.7, 56.5, 56.7, 56.4, 56.5, 56.3, 56.2, 56.3, 56.6, 56.1, 56.0, 56.1, 56.0, 55.7

alpha=0.5, 5 local epochs
9.4, 12.9, 39.4, 47.3, 51.6, 52.9, 54.3, 54.0, 53.9, 54.2, 54.0, 53.9, 54.3, 53.8, 53.7, 53.5, 53.1, 53.5, 53.2, 52.8, 53.1, 53.0, 52.7, 52.4, 52.5, 52.0, 51.9, 51.9, 52.2, 52.0, 51.7

alpha=0.1, 5 local epochs
9.4, 10.0, 21.8, 30.5, 37.2, 39.5, 42.4, 44.3, 45.0, 45.4, 43.4, 44.0, 43.6, 43.8, 43.5, 43.2, 43.5, 42.8, 43.2, 42.4, 42.0, 43.0, 42.5, 42.8, 43.3, 42.8, 42.1, 43.2, 42.2, 43.0, 42.8

More local training per round converges much faster early but reaches a lower ceiling, and the
penalty grows with heterogeneity: -4.3 points at alpha=10, -4.9 at alpha=0.5, -6.6 at alpha=0.1.

All three 5-epoch runs show the same signature: a rapid climb, an early peak, then a slow decline
while training loss keeps falling and evaluation loss rises. At alpha=0.1 training loss reaches 0.094
while evaluation loss reaches 2.89, above the untrained model's 2.30. Clients fit their local label
distributions increasingly well and the averaged global model generalises worse.

Note this compares equal round counts, not equal total local computation: the 5-epoch runs perform
five times more local work overall.

## Methodology note

An earlier set of runs was discarded. Flower's stored simulation config silently reverted from 10
SuperNodes to the default 2 partway through the session, so client count was not controlled. All runs
above pass it explicitly on the command line.

A controlled check at alpha=10, lr=0.1, seed 1 gives 22.7 with 2 clients versus 29.2 with 10, so
federation size alone accounts for several points: averaging over ten updates dilutes individual
client drift, where with two clients each client's drift is half the global update.

## Environment note

charset-normalizer is pinned below 3.5.1 in pyproject.toml. Version 3.5.1 was published mid-session
with no Windows wheel, which broke Flower's per-run dependency install (exit code 608).