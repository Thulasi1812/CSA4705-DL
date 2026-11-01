# Q4: Optimizer Comparison
# CSA47 - Deep Learning Assignment 2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

output_folder = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# 1. Load and prepare dataset
# ----------------------------------------

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

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

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

# ----------------------------------------
# 2. Define neural network
# ----------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(X_train.shape[1], 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------
# 3. Train using different optimizers
# ----------------------------------------

optimizers = ["SGD", "Momentum", "Adam"]
results = {}
summary = []

for optimizer_name in optimizers:

    print("\nTraining with", optimizer_name)

    torch.manual_seed(42)
    model = NeuralNetwork()

    criterion = nn.BCEWithLogitsLoss()

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01
        )

    elif optimizer_name == "Momentum":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=0.01, momentum=0.9
        )

    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=0.001
        )

    train_losses = []
    val_losses = []

    for epoch in range(25):

        model.train()
        total_loss = 0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            output = model(batch_X)
            loss = criterion(output, batch_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_losses.append(total_loss / len(train_loader))

        model.eval()

        with torch.no_grad():
            val_output = model(X_val)
            val_loss = criterion(val_output, y_val).item()

        val_losses.append(val_loss)

        print(
            "Epoch", epoch + 1,
            "Train Loss:", round(train_losses[-1], 4),
            "Validation Loss:", round(val_loss, 4)
        )

    # Test accuracy
    model.eval()

    with torch.no_grad():
        predictions = torch.sigmoid(model(X_test))
        predictions = (predictions >= 0.5).float()

    test_accuracy = accuracy_score(
        y_test.numpy(),
        predictions.numpy()
    )

    results[optimizer_name] = {
        "train_loss": train_losses,
        "val_loss": val_losses
    }

    summary.append({
        "Optimizer": optimizer_name,
        "Test Accuracy": test_accuracy,
        "Final Training Loss": train_losses[-1],
        "Final Validation Loss": val_losses[-1]
    })


# ----------------------------------------
# 4. Save results
# ----------------------------------------

results_df = pd.DataFrame(summary)

print("\nOptimizer Comparison Results")
print(results_df.to_string(index=False))

results_df.to_csv(
    os.path.join(output_folder, "Q4_results.csv"),
    index=False
)


# ----------------------------------------
# 5. Image 1: Loss comparison graph
# ----------------------------------------

plt.figure(figsize=(9, 5))

for name in optimizers:
    plt.plot(
        results[name]["val_loss"],
        label=name + " Validation Loss"
    )

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Optimizer Validation Loss Comparison")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_folder, "Q4_loss_graph.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------
# 6. Image 2: Accuracy comparison graph
# ----------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    results_df["Optimizer"],
    results_df["Test Accuracy"] * 100
)

plt.xlabel("Optimizer")
plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.grid(axis="y")

plt.savefig(
    os.path.join(output_folder, "Q4_accuracy_graph.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nQ4 completed.")
print("Results and graphs saved in:", output_folder)