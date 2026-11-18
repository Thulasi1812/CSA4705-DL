"""Experiment 14: Gradient descent on a quadratic cost function."""
# Minimize f(w) = (w - 5)^2 + 3. Its derivative is 2(w - 5).
learning_rate = 0.1
w = -4.0
iterations = 50

for step in range(iterations):
    gradient = 2 * (w - 5)
    w -= learning_rate * gradient

print(f"Estimated minimum parameter w = {w:.6f}")
print(f"Minimum cost f(w) = {(w - 5)**2 + 3:.6f}")
print("The theoretical minimum is at w=5 with cost=3.")
