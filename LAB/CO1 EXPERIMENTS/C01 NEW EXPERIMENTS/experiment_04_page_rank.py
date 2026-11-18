"""Experiment 4: Eigenvalues/eigenvectors of a web-page transition matrix."""
import numpy as np

# Columns represent the page being visited from; each column sums to 1.
P = np.array([
    [0.0, 0.5, 1.0],
    [0.5, 0.0, 0.0],
    [0.5, 0.5, 0.0]
])

values, vectors = np.linalg.eig(P)
idx = np.argmin(np.abs(values - 1))
importance = np.abs(np.real(vectors[:, idx]))
importance /= importance.sum()

print("Transition matrix:\n", P)
print("Eigenvalues:\n", values)
print("Eigenvector associated with eigenvalue closest to 1:\n", vectors[:, idx])
print("Normalized page importance:")
for i, score in enumerate(importance, start=1):
    print(f"Page {i}: {score:.4f}")
