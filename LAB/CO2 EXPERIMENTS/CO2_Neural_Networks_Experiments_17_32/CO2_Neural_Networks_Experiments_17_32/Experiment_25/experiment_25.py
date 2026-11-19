# XOR classification with hidden-layer neural network
import numpy as np, matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
X=np.array([[0,0],[0,1],[1,0],[1,1]]);y=np.array([0,1,1,0])
m=MLPClassifier(hidden_layer_sizes=(4,),activation="tanh",solver="lbfgs",max_iter=2000,random_state=3).fit(X,y)
print("Predicted XOR:",m.predict(X))
xx,yy=np.meshgrid(np.linspace(-.2,1.2,150),np.linspace(-.2,1.2,150));z=m.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
plt.contourf(xx,yy,z,alpha=.3);plt.scatter(X[:,0],X[:,1],c=y,edgecolors="black");plt.title("XOR decision regions");plt.tight_layout();plt.savefig("output.png")
