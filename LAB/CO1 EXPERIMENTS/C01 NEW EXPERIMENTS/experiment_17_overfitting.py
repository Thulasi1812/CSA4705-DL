"""Experiment 17: Demonstrate and analyse overfitting in a neural network."""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# A large network trained for many iterations can overfit this small dataset.
model = MLPClassifier(hidden_layer_sizes=(100, 100), activation="relu",
                      solver="adam", alpha=0.00001, max_iter=1500,
                      random_state=42)
model.fit(X_train, y_train)
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print(f"Training accuracy: {train_acc:.4f}")
print(f"Testing accuracy:  {test_acc:.4f}")
print(f"Training error: {1-train_acc:.4f}")
print(f"Testing error:  {1-test_acc:.4f}")
if train_acc - test_acc > 0.10:
    print("Possible overfitting: training accuracy is much higher than testing accuracy.")
else:
    print("The train-test gap is not large in this run; inspect learning curves and repeat with other splits.")
print("Improvements: use more data, regularization (higher alpha), early stopping, dropout, or a smaller network.")

plt.bar(["Training", "Testing"], [train_acc, test_acc])
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")
plt.tight_layout()
plt.show()
