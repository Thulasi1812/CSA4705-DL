"""Experiment 10: Generate random temperatures and analyse their distribution."""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
temperatures = rng.normal(loc=25, scale=4, size=1000)

print("Sample size:", len(temperatures))
print("Sample mean:", temperatures.mean())
print("Sample standard deviation:", temperatures.std(ddof=1))
print("Expected normal mean=25 and standard deviation=4.")
plt.hist(temperatures, bins=25, density=True, alpha=0.7, edgecolor="black")
x = np.linspace(temperatures.min(), temperatures.max(), 300)
pdf = (1 / (4 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - 25) / 4) ** 2)
plt.plot(x, pdf, linewidth=2)
plt.title("Random Temperature Observations")
plt.xlabel("Temperature (°C)")
plt.ylabel("Density")
plt.tight_layout()
plt.show()
