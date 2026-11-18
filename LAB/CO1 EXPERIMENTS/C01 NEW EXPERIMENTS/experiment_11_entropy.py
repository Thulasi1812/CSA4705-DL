"""Experiment 11: Entropy of a loan applicant classification dataset."""
import math
from collections import Counter

labels = ["Low", "Low", "High", "Low", "High", "Low", "High", "Low", "Low", "High"]
counts = Counter(labels)
n = len(labels)
entropy = -sum((count/n) * math.log2(count/n) for count in counts.values())

print("Class counts:", dict(counts))
print(f"Dataset entropy: {entropy:.4f} bits")
print("Entropy is 0 when all applicants belong to one class; higher entropy means greater uncertainty.")
