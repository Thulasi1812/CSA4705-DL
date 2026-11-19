# Experiment 19: L1 and L2 regularization
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.metrics import mean_squared_error
X,y=make_regression(n_samples=300,n_features=12,n_informative=5,noise=20,random_state=42)
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.25,random_state=42)
models={"OLS":LinearRegression(),"L1 (Lasso)":Lasso(alpha=5),"L2 (Ridge)":Ridge(alpha=5)}
coef=[]; losses=[]
for n,m in models.items():
 p=make_pipeline(StandardScaler(),m).fit(Xt,yt); coef.append(p[-1].coef_); losses.append(mean_squared_error(yv,p.predict(Xv)))
fig,ax=plt.subplots(); ax.bar(np.arange(12)-.25,coef[0],.25,label="OLS");ax.bar(np.arange(12),coef[1],.25,label="L1");ax.bar(np.arange(12)+.25,coef[2],.25,label="L2");ax.set_title("Regularization effects on coefficients");ax.set_xlabel("Feature");ax.legend();fig.tight_layout();fig.savefig("output.png")
print("Test MSE:",dict(zip(models,losses)))
