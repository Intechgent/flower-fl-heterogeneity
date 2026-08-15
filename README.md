# Federated Learning under Data Heterogeneity

Measuring how FedAvg behaves as client data becomes non-IID, using Flower with Dirichlet partitioning on CIFAR-10.

**Status: work in progress.**

## Setup

- Flower 1.33, PyTorch quickstart template as the starting scaffold
- CIFAR-10, small CNN, FedAvg
- 10 simulated clients, 30 rounds, 1 local epoch per round
- SGD, lr=0.1, momentum=0.9
- Dirichlet partitioning by label, alpha in {10, 0.5, 0.1}, run at two seeds
- Metric: centralised accuracy on the full CIFAR-10 test set, evaluated server-side each round

## Results

Final accuracy after 30 rounds:

| alpha | seed 42 | seed 1 |
|---|---|---|
| 10 (mild skew) | 16.7 (diverged) | 22.7 |
| 0.5 (moderate) | 24.1 | 15.9 |
| 0.1 (severe) | 18.3 | 18.1 |

Full per-round curves are in [results/raw_results.md](results/raw_results.md).

## What the runs show

**Severe heterogeneity is consistent.** At alpha=0.1 both seeds land in the same place (18.3 and 18.1) and show the same behaviour: oscillation between roughly 16 and 21 percent with no stable convergence, plus occasional sharp drops.

**Mild and moderate heterogeneity are not.** At alpha=0.5 the two seeds differ by 8 points: one converged smoothly to about 24 percent, the other never converged and finished at 15.9. At alpha=10 one seed trained normally while the other diverged completely at round 5, sitting at random-guessing accuracy with train loss pinned at ln(10) for fifteen rounds before partially recovering.

**The variance is larger than the effect.** With only two seeds, between-seed differences at a fixed alpha are bigger than the differences between alpha values, so no clean alpha-to-performance relationship can be claimed from these runs.

## Open question

Is the instability caused by the learning rate rather than the heterogeneity? lr=0.1 with momentum 0.9 is aggressive for this model, and a diverged run at alpha=10 points that way. If training were stable, the alpha effect might separate cleanly instead of being swamped by seed variance. Next step is to rerun the sweep at a lower learning rate.

## Running it

Install dependencies, set the client count, then run:

    pip install -r requirements.txt
    flwr federation simulation-config --num-supernodes 10
    flwr run . --stream

Alpha is set in `pytorchexample/task.py` (the `DirichletPartitioner` call); rounds and learning rate are in `pyproject.toml`.

Built on Flower's PyTorch quickstart template.