"""Experiment 1: Vector addition, subtraction and scalar multiplication."""
import numpy as np

position = np.array([2, 3, 1], dtype=float)
movement = np.array([4, -1, 2], dtype=float)
scale = 2

print("Initial position:", position)
print("Movement vector:", movement)
print("New position (addition):", position + movement)
print("Relative displacement (subtraction):", position - movement)
print("Scaled movement:", scale * movement)
