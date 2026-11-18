"""Experiment 12: Binary cross-entropy loss for spam predictions."""
import numpy as np

# 1 = spam, 0 = not spam
actual = np.array([1, 0, 1, 1, 0])
predicted = np.array([0.90, 0.20, 0.60, 0.80, 0.10])
eps = 1e-15
p = np.clip(predicted, eps, 1 - eps)
losses = -(actual * np.log(p) + (1 - actual) * np.log(1 - p))

print("Actual labels:", actual)
print("Predicted probabilities:", predicted)
print("Per-example cross-entropy:", np.round(losses, 4))
print("Mean binary cross-entropy:", round(float(losses.mean()), 4))
print("Lower loss indicates better-calibrated, more accurate probability predictions.")
