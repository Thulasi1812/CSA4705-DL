# Q2: Manual Backpropagation and Gradient Checking
# CSA47 - Deep Learning Assignment 2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

output_folder = os.path.dirname(os.path.abspath(__file__))

# 1. Sample dataset
X = np.array([
    [0.1, 0.2],
    [0.3, 0.7],
    [0.8, 0.4],
    [0.9, 0.6]
])

y = np.array([[0], [0], [1], [1]])

# 2. Initialize network parameters
W1 = np.random.randn(2, 3) * 0.5
b1 = np.zeros((1, 3))
W2 = np.random.randn(3, 1) * 0.5
b2 = np.zeros((1, 1))


# 3. Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# 4. Forward propagation
def forward():
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    prediction = sigmoid(z2)

    prediction = np.clip(prediction, 1e-8, 1 - 1e-8)

    loss = -np.mean(
        y * np.log(prediction) +
        (1 - y) * np.log(1 - prediction)
    )

    return loss, a1, prediction


# 5. Manual backpropagation
def backward():
    loss, a1, prediction = forward()

    m = len(X)

    dz2 = (prediction - y) / m
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * a1 * (1 - a1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    gradients = {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

    return loss, gradients


# 6. Numerical gradient calculation
def numerical_gradient(parameter, epsilon):
    numerical = np.zeros_like(parameter)

    for index in np.ndindex(parameter.shape):
        original = parameter[index]

        parameter[index] = original + epsilon
        loss_plus = forward()[0]

        parameter[index] = original - epsilon
        loss_minus = forward()[0]

        parameter[index] = original

        numerical[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

    return numerical


# 7. Compare gradients
loss, analytical = backward()

parameters = {
    "W1": W1,
    "b1": b1,
    "W2": W2,
    "b2": b2
}

results = []

print("Q2: Manual Backpropagation")
print("Initial Loss:", round(loss, 6))

for epsilon in [1e-5, 1e-7]:
    print("\nEpsilon:", epsilon)

    for name, parameter in parameters.items():
        numerical = numerical_gradient(parameter, epsilon)

        difference = np.linalg.norm(
            analytical[name] - numerical
        )

        results.append({
            "Parameter": name,
            "Epsilon": epsilon,
            "Gradient Difference": difference
        })

        print(name, "Difference:", f"{difference:.10f}")


# 8. Bug detection demonstration
correct_gradient = analytical["W2"]
numerical = numerical_gradient(W2, 1e-5)

wrong_gradient = correct_gradient + 0.1

correct_difference = np.linalg.norm(
    correct_gradient - numerical
)

wrong_difference = np.linalg.norm(
    wrong_gradient - numerical
)

print("\nBug Detection")
print("Correct gradient difference:", correct_difference)
print("Incorrect gradient difference:", wrong_difference)

if wrong_difference > 1e-4:
    print("Gradient check detected the bug.")


# 9. Save results
results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(output_folder, "Q2_gradient_check.csv"),
    index=False
)


# 10. Create graph image
plt.figure(figsize=(8, 5))

for name in parameters:
    subset = results_df[results_df["Parameter"] == name]

    plt.plot(
        subset["Epsilon"].astype(str),
        subset["Gradient Difference"],
        marker="o",
        label=name
    )

plt.xlabel("Epsilon")
plt.ylabel("Gradient Difference")
plt.title("Manual Backpropagation Gradient Check")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_folder, "Q2_gradient_check.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nQ2 completed.")
print("Results and graph saved in:", output_folder)