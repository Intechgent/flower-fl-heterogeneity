# Federated Learning under Data Heterogeneity

Measuring how FedAvg degrades as client data becomes non-IID, using Flower with Dirichlet partitioning on CIFAR-10.

**Status: work in progress.**

## Setup so far

- Flower 1.33, PyTorch quickstart template as the starting scaffold
- CIFAR-10, small CNN, FedAvg, 10 rounds
- 10 simulated clients (IID partitioning so far)

## Early observations (IID baseline)

Varying the number of clients with IID data:

| clients | accuracy @ round 10 | curve |
|---|---|---|
| 2 | 20.9% | noisy, plateaus early |
| 10 | 26.7% | smoother, still climbing |

With IID partitioning, more clients helped: each client's data is representative, so their updates point in similar directions and averaging 10 of them reduces noise.

The open question is whether this still holds under non-IID data, where client updates should diverge. That is what the Dirichlet experiments will test.

## Next

- Fix random seed for reproducible comparisons
- Swap IidPartitioner for DirichletPartitioner
- Run at α = 10, 0.5, 0.1 and plot accuracy vs rounds

Built on Flower's PyTorch quickstart template.