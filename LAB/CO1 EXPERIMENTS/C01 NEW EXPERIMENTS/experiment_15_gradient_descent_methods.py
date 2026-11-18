"""Experiment 15: Compare batch, stochastic and mini-batch gradient descent."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
X = rng.uniform(-2, 2, 100)
y = 3 * X + 2 + rng.normal(0, 0.3, size=len(X))
X = X.reshape(-1, 1)
Xb = np.c_[np.ones(len(X)), X]
target = np.array([2.0, 3.0])

def loss(theta):
    return np.mean((Xb @ theta - y) ** 2)

def train(method, epochs=40, lr=0.05, batch_size=10):
    theta = np.zeros(2)
    history = []
    for _ in range(epochs):
        if method == "batch":
            batches = [np.arange(len(y))]
        else:
            order = rng.permutation(len(y))
            size = 1 if method == "stochastic" else batch_size
            batches = [order[i:i+size] for i in range(0, len(y), size)]
        for idx in batches:
            error = Xb[idx] @ theta - y[idx]
            grad = 2 * (Xb[idx].T @ error) / len(idx)
            theta -= lr * grad
        history.append(loss(theta))
    return theta, history

results = {}
for method in ("batch", "stochastic", "mini-batch"):
    theta, history = train(method)
    results[method] = history
    print(f"{method}: parameters={np.round(theta, 3)}, final MSE={history[-1]:.4f}")

for method, history in results.items():
    plt.plot(history, label=method)
plt.xlabel("Epoch")
plt.ylabel("Mean squared error")
plt.title("Gradient Descent Convergence")
plt.legend()
plt.tight_layout()
plt.show()
