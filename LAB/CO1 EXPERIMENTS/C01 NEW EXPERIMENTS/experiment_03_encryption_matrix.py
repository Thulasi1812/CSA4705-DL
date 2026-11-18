"""Experiment 3: Determinant, inverse and rank of an encryption matrix."""
import numpy as np

key = np.array([[3, 3], [2, 5]], dtype=float)
det = np.linalg.det(key)
rank = np.linalg.matrix_rank(key)

print("Encryption matrix:\n", key)
print("Determinant:", round(det, 4))
print("Rank:", rank)
if not np.isclose(det, 0):
    inverse = np.linalg.inv(key)
    print("Inverse:\n", inverse)
    print("Suitable as an invertible real-valued matrix:", rank == key.shape[0])
else:
    print("Matrix is singular; no inverse exists, so it is unsuitable for reversible matrix encryption.")
print("Note: practical Hill-cipher encryption requires an inverse modulo the chosen alphabet size.")
