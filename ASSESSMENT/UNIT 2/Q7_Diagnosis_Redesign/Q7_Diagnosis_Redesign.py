# Q7: Diagnose, Redesign and Validate
# CSA47 - Deep Learning Assignment 2

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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

input_size = X_train.shape[1]


# ----------------------------------------
# 2. Define baseline and redesigned models
# ----------------------------------------

class BaselineModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)


class RedesignedModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------
# 3. Training function
# ----------------------------------------

def train_model(model, use_regularization):

    criterion = nn.BCEWithLogitsLoss()

    if use_regularization:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=0.001
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

    return train_losses, val_losses


# ----------------------------------------
# 4. Train both models
# ----------------------------------------

print("Training baseline model...")
baseline = BaselineModel()

baseline_train, baseline_val = train_model(
    baseline, use_regularization=False
)

print("Training redesigned model...")
redesigned = RedesignedModel()

redesigned_train, redesigned_val = train_model(
    redesigned, use_regularization=True
)


# ----------------------------------------
# 5. Evaluate test accuracy
# ----------------------------------------

def test_accuracy(model):

    model.eval()

    with torch.no_grad():
        output = torch.sigmoid(model(X_test))
        predictions = (output >= 0.5).float()

    return accuracy_score(
        y_test.numpy(),
        predictions.numpy()
    )


baseline_accuracy = test_accuracy(baseline)
redesigned_accuracy = test_accuracy(redesigned)

results = pd.DataFrame([
    {
        "Model": "Baseline",
        "Test Accuracy": baseline_accuracy
    },
    {
        "Model": "Redesigned",
        "Test Accuracy": redesigned_accuracy
    }
])

print("\nTest Results")
print(results.to_string(index=False))

results.to_csv(
    os.path.join(output_folder, "Q7_results.csv"),
    index=False
)


# ----------------------------------------
# 6. Image 1: Training and validation loss
# ----------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(baseline_train, label="Baseline Training Loss")
plt.plot(baseline_val, label="Baseline Validation Loss")
plt.plot(redesigned_train, label="Redesigned Training Loss")
plt.plot(redesigned_val, label="Redesigned Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Baseline vs Redesigned Model")
plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(output_folder, "Q7_loss_graph.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------
# 7. Image 2: Test accuracy comparison
# ----------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    results["Model"],
    results["Test Accuracy"] * 100
)

plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.title("Model Validation Comparison")
plt.grid(axis="y")

plt.savefig(
    os.path.join(output_folder, "Q7_accuracy_graph.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------
# 8. Print conclusion
# ----------------------------------------

print("\nDiagnosis:")
print("The baseline model has more layers and parameters.")
print("The redesigned model uses fewer layers, dropout and L2 regularization.")

print("\nValidation:")
if redesigned_accuracy >= baseline_accuracy:
    print("The redesigned model achieved equal or better test accuracy.")
else:
    print("The baseline model achieved higher test accuracy.")
    print("Compare the loss graphs before deciding which model generalizes better.")

print("\nQ7 completed.")
print("Results and graphs saved in:", output_folder)