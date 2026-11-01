# Q1: Activation Functions and Network Depth
# CSA47 - Deep Learning Assignment 2

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Output folder
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load built-in dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split dataset: 60% train, 20% validation, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Normalize data using training data only
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
y_val = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)


# Neural network model
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, activation, depth):
        super().__init__()

        layers = []
        current_size = input_size

        for _ in range(depth):
            layers.append(nn.Linear(current_size, 32))

            if activation == "Sigmoid":
                layers.append(nn.Sigmoid())
            elif activation == "Tanh":
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())

            current_size = 32

        layers.append(nn.Linear(current_size, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# Train and evaluate
results = []
loss_graphs = {}

experiments = [
    ("Sigmoid", 2),
    ("Tanh", 2),
    ("ReLU", 2),
    ("Sigmoid", 6),
    ("Tanh", 6),
    ("ReLU", 6)
]

for activation, depth in experiments:

    name = activation + "_" + str(depth) + "_Layers"
    print("\nRunning:", name)

    model = NeuralNetwork(X_train.shape[1], activation, depth)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []

    for epoch in range(15):

        model.train()
        optimizer.zero_grad()

        output = model(X_train)
        loss = criterion(output, y_train)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val)

        val_losses.append(val_loss.item())

    # Test accuracy
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        predictions = (torch.sigmoid(test_output) >= 0.5).int()

    accuracy = accuracy_score(
        y_test.numpy(),
        predictions.numpy()
    )

    results.append({
        "Activation": activation,
        "Hidden Layers": depth,
        "Test Accuracy": accuracy
    })

    loss_graphs[name] = (train_losses, val_losses)

    print("Test Accuracy:", round(accuracy * 100, 2), "%")


# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(OUTPUT_DIR, "Q1_results.csv"),
    index=False
)

print("\nFinal Results:")
print(results_df)


# Graph 1: Test accuracy comparison
plt.figure(figsize=(9, 5))

labels = [
    row["Activation"] + "\n" + str(row["Hidden Layers"]) + " layers"
    for row in results
]

accuracies = [
    row["Test Accuracy"] * 100
    for row in results
]

plt.bar(labels, accuracies)
plt.xlabel("Activation Function and Network Depth")
plt.ylabel("Test Accuracy (%)")
plt.title("Q1: Activation Function and Depth Comparison")
plt.ylim(0, 100)
plt.grid(axis="y")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Q1_accuracy_graph.png")
)
plt.close()


# Graph 2: Training and validation loss
plt.figure(figsize=(10, 6))

for name, (train_loss, val_loss) in loss_graphs.items():
    plt.plot(train_loss, label=name + " Train")
    plt.plot(val_loss, linestyle="--", label=name + " Validation")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Q1: Training and Validation Loss")
plt.legend(fontsize=7)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "Q1_loss_graph.png")
)
plt.close()


print("\nGraphs and results saved in:")
print(OUTPUT_DIR)
print("Q1 completed.")