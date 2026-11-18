"""Experiment 7: Mean, variance and standard deviation of student marks."""
import numpy as np

marks = np.array([78, 85, 69, 92, 88, 74, 81, 90], dtype=float)
print("Marks:", marks)
print("Mean:", np.mean(marks))
print("Population variance:", np.var(marks))
print("Population standard deviation:", np.std(marks))
print("Interpretation: a smaller standard deviation means marks are more consistent.")
