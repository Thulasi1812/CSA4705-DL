"""Experiment 6: Euclidean and Manhattan distance comparison."""
import numpy as np

customers = {
    "A": np.array([2, 3, 5]),
    "B": np.array([4, 1, 6]),
    "C": np.array([8, 2, 1]),
}
def euclidean(a, b):
    return float(np.linalg.norm(a - b))

def manhattan(a, b):
    return float(np.abs(a - b).sum())

for name in ("B", "C"):
    a, b = customers["A"], customers[name]
    print(f"A to {name}: Euclidean={euclidean(a,b):.3f}, Manhattan={manhattan(a,b):.3f}")
print("Euclidean measures straight-line distance; Manhattan sums feature-wise differences.")
print("Choose based on feature scale, geometry, and sensitivity to large differences.")
