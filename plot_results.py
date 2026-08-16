import numpy as np
import matplotlib.pyplot as plt

rounds = list(range(0, 31))

alpha_10 = [9.4, 17.7, 29.7, 35.3, 40.1, 42.7, 44.9, 46.4, 48.1, 49.6, 50.6,
            51.6, 52.3, 53.5, 53.9, 54.9, 55.4, 56.0, 56.5, 56.8, 57.4,
            57.7, 58.0, 58.3, 58.7, 58.8, 59.1, 59.6, 60.0, 60.0, 60.0]

alpha_05 = [9.4, 11.0, 24.7, 31.7, 35.4, 38.4, 40.5, 42.8, 44.4, 45.6, 46.8,
            47.8, 48.8, 49.5, 50.5, 51.4, 52.6, 53.0, 53.7, 54.1, 54.3,
            54.6, 55.0, 55.1, 55.5, 55.5, 55.6, 56.0, 56.4, 56.4, 56.6]

alpha_01 = [9.4, 10.0, 18.6, 24.9, 27.6, 29.0, 30.5, 33.2, 35.7, 38.3, 40.8,
            42.3, 43.5, 44.7, 45.6, 46.0, 47.1, 47.6, 47.9, 48.2, 48.6,
            48.6, 49.1, 49.4, 49.6, 49.2, 49.3, 49.5, 49.1, 48.8, 49.4]

# Chart 1: accuracy vs rounds for three alpha values
fig, ax = plt.subplots(figsize=(9, 6))

ax.plot(rounds, alpha_10, label="alpha=10 (mild skew)", color="#2166ac", linewidth=2)
ax.plot(rounds, alpha_05, label="alpha=0.5 (moderate skew)", color="#f4a582", linewidth=2)
ax.plot(rounds, alpha_01, label="alpha=0.1 (severe skew)", color="#b2182b", linewidth=2)

ax.set_xlabel("Round")
ax.set_ylabel("Accuracy (%)")
ax.set_title("FedAvg accuracy under Dirichlet partitioning (lr=0.01, 10 clients, 1 local epoch)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("results/alpha_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Chart 2: effect of local training amount, per alpha
alphas = ["10\n(mild)", "0.5\n(moderate)", "0.1\n(severe)"]
epoch_1 = [60.0, 56.6, 49.4]
epoch_5 = [55.7, 51.7, 42.8]

x = np.arange(len(alphas))
width = 0.35

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.bar(x - width/2, epoch_1, width, label="1 local epoch", color="#4393c3")
ax2.bar(x + width/2, epoch_5, width, label="5 local epochs", color="#d6604d")

ax2.set_ylabel("Final accuracy (%)")
ax2.set_xlabel("Heterogeneity")
ax2.set_title("Effect of local training amount on final accuracy (lr=0.01, 10 clients, 30 rounds)")
ax2.set_xticks(x)
ax2.set_xticklabels(alphas)
ax2.legend()
ax2.grid(True, alpha=0.3, axis="y")

for i, (v1, v5) in enumerate(zip(epoch_1, epoch_5)):
    ax2.text(i - width/2, v1 + 0.5, f"{v1}", ha="center", fontsize=9)
    ax2.text(i + width/2, v5 + 0.5, f"{v5}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig("results/local_epochs_comparison.png", dpi=150, bbox_inches="tight")
plt.show()