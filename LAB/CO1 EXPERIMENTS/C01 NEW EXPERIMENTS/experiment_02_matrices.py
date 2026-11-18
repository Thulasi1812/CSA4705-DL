"""Experiment 2: Matrix multiplication and transpose."""
import numpy as np

transform = np.array([[2, 1], [0, 3]])
coordinates = np.array([[1, 2], [3, 4]])

product = transform @ coordinates
print("Transformation matrix:\n", transform)
print("Coordinate matrix:\n", coordinates)
print("Matrix product:\n", product)
print("Transpose of transformation matrix:\n", transform.T)
print("Analysis: multiplication combines the transformations; transpose swaps rows and columns.")
