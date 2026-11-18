"""Experiment 16: Linear regression for house-price prediction."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Features: area (sq ft), bedrooms, location score (1-5)
X = np.array([
    [1000, 2, 3], [1200, 2, 4], [1500, 3, 3], [1800, 3, 5],
    [900, 1, 2], [2000, 4, 5], [1600, 3, 4], [1100, 2, 2],
    [2200, 4, 4], [1400, 3, 2], [1750, 3, 4], [1300, 2, 3]
], dtype=float)
# Illustrative prices in lakh.
y = np.array([45, 55, 65, 88, 35, 100, 78, 42, 95, 50, 82, 52], dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Actual prices:", y_test)
print("Predicted prices:", np.round(pred, 2))
print("MAE:", round(mean_absolute_error(y_test, pred), 2))
print("MSE:", round(mean_squared_error(y_test, pred), 2))
print("R²:", round(r2_score(y_test, pred), 3))
