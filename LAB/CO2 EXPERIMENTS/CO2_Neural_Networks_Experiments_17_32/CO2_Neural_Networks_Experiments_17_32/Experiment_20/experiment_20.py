# Experiment 20: Learning-rate comparison
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
X,y=make_regression(n_samples=300,n_features=10,noise=20,random_state=42); X=StandardScaler().fit_transform(X)
rates=[.001,.01,.1]; fig,ax=plt.subplots()
for lr in rates:
 m=MLPRegressor(hidden_layer_sizes=(40,),learning_rate_init=lr,max_iter=300,random_state=42).fit(X,y)
 ax.plot(m.loss_curve_,label=f"LR={lr}")
ax.set_title("Learning rate and convergence");ax.set_xlabel("Iteration");ax.set_ylabel("Training loss");ax.legend();fig.tight_layout();fig.savefig("output.png")
