# Single layer perceptron for binary classification
import numpy as np, matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
X=np.array([[30,500],[45,650],[25,300],[60,900],[35,700],[55,800]],float);y=np.array([0,1,0,1,0,1])
model=Perceptron(random_state=42).fit(X,y);print("Predictions:",model.predict(X))
plt.scatter(X[:,0],X[:,1],c=y);plt.xlabel("Income (illustrative)");plt.ylabel("Credit score");plt.title("Perceptron: loan classification");plt.tight_layout();plt.savefig("output.png")
