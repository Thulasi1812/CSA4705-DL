# Q5: CNN Architecture Design
# CSA47 - Deep Learning Assignment 2

import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

output_folder = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------
# 1. Define CNN Model
# ----------------------------------------

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ----------------------------------------
# 2. Create model and test input
# ----------------------------------------

model = SimpleCNN()

# Example input: 32x32 RGB image
sample_image = torch.randn(1, 3, 32, 32)

output = model(sample_image)

print("Q5: CNN Architecture Design")
print("--------------------------------")
print("Input image shape:", sample_image.shape)
print("Output shape:", output.shape)
print("Number of classes: 10")


# ----------------------------------------
# 3. Display layer output shapes
# ----------------------------------------

x = sample_image

print("\nLayer Output Shapes:")

for layer in model.features:
    x = layer(x)
    print(layer.__class__.__name__, ":", x.shape)

for layer in model.classifier:
    x = layer(x)
    print(layer.__class__.__name__, ":", x.shape)


# ----------------------------------------
# 4. Calculate parameters
# ----------------------------------------

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\nTotal Parameters:", total_parameters)
print("Trainable Parameters:", trainable_parameters)


# ----------------------------------------
# 5. Save CNN architecture diagram
# ----------------------------------------

layers = [
    "Input Image\n32 x 32 x 3",
    "Conv2D + ReLU\n16 filters",
    "Max Pooling\n16 x 16 x 16",
    "Conv2D + ReLU\n32 filters",
    "Max Pooling\n8 x 8 x 32",
    "Conv2D + ReLU\n64 filters",
    "Max Pooling\n4 x 4 x 64",
    "Flatten",
    "Dense + ReLU\n128 units",
    "Output Layer\n10 classes"
]

plt.figure(figsize=(10, 8))

for i, layer_name in enumerate(layers):
    y_position = len(layers) - i

    plt.text(
        0.5,
        y_position,
        layer_name,
        ha="center",
        va="center",
        fontsize=10,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            edgecolor="black"
        )
    )

    if i < len(layers) - 1:
        plt.annotate(
            "",
            xy=(0.5, y_position - 0.65),
            xytext=(0.5, y_position - 0.25),
            arrowprops=dict(arrowstyle="->")
        )

plt.xlim(0, 1)
plt.ylim(0, len(layers) + 1)
plt.axis("off")
plt.title("CNN Architecture")

plt.savefig(
    os.path.join(output_folder, "Q5_CNN_Architecture.png"),
    dpi=150,
    bbox_inches="tight"
)

plt.show()

print("\nQ5 completed.")
print("Architecture image saved in:", output_folder)