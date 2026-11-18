"""Experiment 5: Dot product and cosine similarity of customer preferences."""
import numpy as np

customers = {
    "Customer A": np.array([5, 2, 0, 1]),
    "Customer B": np.array([4, 1, 0, 2]),
    "Customer C": np.array([0, 1, 5, 4]),
}
names = list(customers)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        a, b = customers[names[i]], customers[names[j]]
        dot = np.dot(a, b)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        cosine = dot / denom if denom else 0.0
        print(f"{names[i]} vs {names[j]}: dot product={dot}, cosine similarity={cosine:.4f}")
print("A cosine similarity closer to 1 indicates more similar preferences.")
