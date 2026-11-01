# Q6: Model Architecture, Capacity and Deployment
# CSA47 - Deep Learning Assignment 2

import os
import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt

output_folder = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# 1. Define different model sizes
# ----------------------------------------

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 32),
            nn.ReLU(),
            nn.Linear(32, 10)
        )

    def forward(self, x):
        return self.model(x)


class MediumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


class LargeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------
# 2. Compare model capacity
# ----------------------------------------

models = {
    "Small": SmallModel(),
    "Medium": MediumModel(),
    "Large": LargeModel()
}

results = []

print("Q6: Model Capacity Comparison")
print("--------------------------------")

for name, model in models.items():

    parameters = sum(
        p.numel() for p in model.parameters()
    )

    model_size_mb = parameters * 4 / (1024 * 1024)

    sample = torch.randn(1, 3, 32, 32)

    with torch.no_grad():
        output = model(sample)

    results.append({
        "Model": name,
        "Parameters": parameters,
        "Model Size (MB)": model_size_mb,
        "Output Shape": str(tuple(output.shape))
    })

    print("\nModel:", name)
    print("Parameters:", parameters)
    print("Estimated Size (MB):", round(model_size_mb, 3))
    print("Output Shape:", output.shape)


# ----------------------------------------
# 3. Save comparison results
# ----------------------------------------

results_df = pd.DataFrame(results)

results_df.to_csv(
    os.path.join(output_folder, "Q6_model_comparison.csv"),
    index=False
)

print("\nComparison Table:")
print(results_df.to_string(index=False))


# ----------------------------------------
# 4. Create model size graph
# ----------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    results_df["Model"],
    results_df["Model Size (MB)"]
)

plt.xlabel("Model")
plt.ylabel("Estimated Model Size (MB)")
plt.title("Model Size Comparison")
plt.grid(axis="y")

plt.savefig(
    os.path.join(output_folder, "Q6_model_size.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()


# ----------------------------------------
# 5. Deployment example
# ----------------------------------------

model = SmallModel()
model.eval()

sample_input = torch.randn(1, 3, 32, 32)

# Export model for deployment
torch.save(
    model.state_dict(),
    os.path.join(output_folder, "Q6_small_model.pth")
)

print("\nDeployment model saved as Q6_small_model.pth")
print("Q6 completed.")