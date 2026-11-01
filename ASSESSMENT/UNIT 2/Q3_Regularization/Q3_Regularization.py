# Q3: Regularization and Overfitting
# CSA47 - Deep Learning Assignment 2

import os
import random
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

# Set random seed
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

output_folder = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# 1. Load dataset
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
# 2. Create neural network
# ----------------------------------------

class NeuralNetwork(nn.Module):

    def __init__(self, use_dropout=False):
        super().__init__()

        if use_dropout:
            self.model = nn.Sequential(
                nn.Linear(X_train.shape[1], 128),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(64, 1)
            )
        else:
            self.model = nn.Sequential(
                nn.Linear(X_train.shape[1], 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------
# 3. Train model
# ----------------------------------------

def train_model(use_regularization):

    model = NeuralNetwork(use_dropout=use_regularization)

    criterion = nn.BCEWithLogitsLoss()

    if use_regularization:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.01
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

    train_losses = []
    val_losses = []

    for epoch in range(30):

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

    # Calculate train and test accuracy
    model.eval()

    with torch.no_grad():
        train_predictions = torch.sigmoid(model(X_train))
        test_predictions = torch.sigmoid(model(X_test))

        train_predictions = (train_predictions >= 0.5).float()
        test_predictions = (test_predictions >= 0.5).float()

    train_accuracy = accuracy_score(
        y_train.numpy(),
        train_predictions.numpy()
    )

    test_accuracy = accuracy_score(
        y_test.numpy(),
        test_predictions.numpy()
    )

    return train_losses, val_losses, train_accuracy, test_accuracy


# ----------------------------------------
# 4. Compare regularization methods
# ----------------------------------------

print("Training model without regularization...")
normal_train, normal_val, normal_train_acc, normal_test_acc = train_model(False)

print("Training model with Dropout and L2 regularization...")
reg_train, reg_val, reg_train_acc, reg_test_acc = train_model(True)

results = pd.DataFrame([
    {
        "Model": "Without Regularization",
        "Training Accuracy": normal_train_acc,
        "Test Accuracy": normal_test_acc
    },
    {
        "Model": "Dropout + L2 Regularization",
        "Training Accuracy": reg_train_acc,
        "Test Accuracy": reg_test_acc
    }
])

print("\nResults:")
print(results.to_string(index=False))

results.to_csv(
    os.path.join(output_folder, "Q3_results.csv"),
    index=False
)

# ----------------------------------------
# 5. Save loss graph image
# ----------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(normal_train, label="Normal Training Loss")
plt.plot(normal_val, label="Normal Validation Loss")
plt.plot(reg_train, label="Regularized Training Loss")
plt.plot(reg_val, label="Regularized Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Overfitting and Regularization")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_folder, "Q3_loss_graph.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

# ----------------------------------------
# 6. Save accuracy graph image
# ----------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    results["Model"],
    results["Test Accuracy"] * 100
)

plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Before and After Regularization")
plt.xticks(rotation=10)
plt.grid(axis="y")

plt.savefig(
    os.path.join(output_folder, "Q3_accuracy_graph.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nQ3 completed.")
print("Results and graphs saved in:", output_folder)