"""Experiment 13: KL divergence between actual and predicted word distributions."""
import numpy as np

actual = np.array([0.50, 0.30, 0.20])
predicted = np.array([0.40, 0.40, 0.20])
eps = 1e-12
p = np.clip(actual, eps, 1)
q = np.clip(predicted, eps, 1)
kl = np.sum(p * np.log(p / q))  # nats

print("Actual distribution:", actual)
print("Predicted distribution:", predicted)
print(f"KL divergence D_KL(P || Q): {kl:.6f} nats")
print("KL divergence is non-negative; 0 means the distributions are identical.")
print("It measures the extra information/coding cost when Q is used instead of P.")
