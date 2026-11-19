# MLP classification on Iris
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import ConfusionMatrixDisplay,accuracy_score
X,y=load_iris(return_X_y=True);a,b,c,d=train_test_split(X,y,test_size=.25,random_state=42,stratify=y)
m=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(16,8),max_iter=800,random_state=42)).fit(a,c);p=m.predict(b);print("Accuracy:",accuracy_score(d,p))
ConfusionMatrixDisplay.from_predictions(d,p);plt.title("Iris MLP confusion matrix");plt.tight_layout();plt.savefig("output.png")
