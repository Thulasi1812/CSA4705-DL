# Experiment 18: Polynomial Regression and Bias-Variance
import numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
rng=np.random.default_rng(42); X=np.linspace(-3,3,80).reshape(-1,1); y=np.sin(X[:,0])+rng.normal(0,.25,80)
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.3,random_state=42)
degrees=[1,3,10]; scores=[]
fig,ax=plt.subplots()
ax.scatter(Xt,yt,label="Training samples",s=18)
xx=np.linspace(-3,3,200).reshape(-1,1)
for d in degrees:
 p=make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(Xt,yt); scores.append((d,mean_squared_error(yt,p.predict(Xt)),mean_squared_error(yv,p.predict(Xv)))); ax.plot(xx,p.predict(xx),label=f"Degree {d}")
ax.legend();ax.set_title("Polynomial degree comparison");ax.set_xlabel("Feature");ax.set_ylabel("Target");fig.tight_layout();fig.savefig("output.png")
print("degree, train MSE, test MSE:",scores)
