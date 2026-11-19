# Experiment 17: Analyse Overfitting
import numpy as np, matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X,y=make_classification(n_samples=500,n_features=20,n_informative=5,n_redundant=5,random_state=42)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=42)
models={"Large network (baseline)":MLPClassifier(hidden_layer_sizes=(100,100),max_iter=500,random_state=42),"Regularized network":MLPClassifier(hidden_layer_sizes=(30,),alpha=.01,max_iter=500,random_state=42)}
results={}
for name,m in models.items():
    pipe=make_pipeline(StandardScaler(),m); pipe.fit(Xtr,ytr)
    results[name]=(accuracy_score(ytr,pipe.predict(Xtr)),accuracy_score(yte,pipe.predict(Xte)))
fig,ax=plt.subplots(); names=list(results); ax.bar(np.arange(2)-.15,[results[n][0] for n in names],.3,label="Training accuracy");ax.bar(np.arange(2)+.15,[results[n][1] for n in names],.3,label="Testing accuracy");ax.set_xticks(range(2),names,rotation=15);ax.set_ylim(0,1);ax.set_ylabel("Accuracy");ax.set_title("Overfitting analysis");ax.legend();fig.tight_layout();fig.savefig("output.png")
print(results)
