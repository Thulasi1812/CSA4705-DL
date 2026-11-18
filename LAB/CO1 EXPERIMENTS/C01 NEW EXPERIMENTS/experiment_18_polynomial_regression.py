"""Experiment 18: Compare polynomial regression degrees and bias-variance trade-off."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(12)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X[:, 0]**2 + X[:, 0] + 2 + rng.normal(0, 1.2, size=100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

degrees = [1, 2, 5, 12]
train_errors, test_errors = [], []
models = []
for degree in degrees:
    model = make_pipeline(
        PolynomialFeatures(degree=degree, include_bias=False),
        StandardScaler(),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test, model.predict(X_test)))
    models.append(model)
    print(f"Degree {degree:2d}: training MSE={train_errors[-1]:.3f}, testing MSE={test_errors[-1]:.3f}")

best = degrees[int(np.argmin(test_errors))]
print(f"Lowest testing error in this run: degree {best}")
print("Low-degree models may underfit (high bias); very high-degree models may overfit (high variance).")

plt.plot(degrees, train_errors, marker="o", label="Training MSE")
plt.plot(degrees, test_errors, marker="o", label="Testing MSE")
plt.xlabel("Polynomial degree")
plt.ylabel("Mean squared error")
plt.title("Polynomial Regression: Bias-Variance Comparison")
plt.legend()
plt.tight_layout()
plt.show()
