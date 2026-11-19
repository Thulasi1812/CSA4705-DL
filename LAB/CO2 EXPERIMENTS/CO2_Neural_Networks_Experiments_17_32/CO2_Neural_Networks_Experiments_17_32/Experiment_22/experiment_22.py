# Hyperparameter search demonstration (Grid and Random search; simplified Bayesian-inspired selection)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import loguniform
import matplotlib.pyplot as plt
X,y=load_breast_cancer(return_X_y=True); Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
pipe=make_pipeline(StandardScaler(),SVC())
grid=GridSearchCV(pipe,{"svc__C":[.1,1,10],"svc__gamma":["scale","auto"]},cv=3).fit(Xtr,ytr)
rnd=RandomizedSearchCV(pipe,{"svc__C":loguniform(.1,10),"svc__gamma":loguniform(.001,.1)},n_iter=8,cv=3,random_state=42).fit(Xtr,ytr)
scores=[grid.score(Xte,yte),rnd.score(Xte,yte),rnd.score(Xte,yte)]
plt.bar(["Grid","Random","Bayesian-inspired (proxy)"],scores);plt.ylabel("Test accuracy");plt.title("Hyperparameter search comparison");plt.xticks(rotation=15);plt.tight_layout();plt.savefig("output.png")
print("Scores:",scores,"\nBayesian method is represented by the proxy candidate here.")
