# Visualize activation functions
import numpy as np, matplotlib.pyplot as plt
x=np.linspace(-6,6,300);plt.plot(x,1/(1+np.exp(-x)),label="Sigmoid");plt.plot(x,np.tanh(x),label="Tanh");plt.plot(x,np.maximum(0,x),label="ReLU");plt.legend();plt.title("Activation functions");plt.xlabel("Input");plt.ylabel("Output");plt.tight_layout();plt.savefig("output.png")
