# Experiment 21: Floating-point precision
import numpy as np, matplotlib.pyplot as plt
values=[np.float16,np.float32,np.float64]; labels=[]; errors=[]
for dtype in values:
 a=dtype(1.0); b=dtype(1e-4); c=dtype(1e-4)
 result=dtype(a+b)-a
 labels.append(dtype.__name__); errors.append(abs(float(result)-1e-4))
print("Computed increment:",[float(dtype(dtype(1)+dtype(1e-4))-dtype(1)) for dtype in values])
fig,ax=plt.subplots();ax.bar(labels,errors);ax.set_title("Floating point error for small increment");ax.set_ylabel("Absolute error");fig.tight_layout();fig.savefig("output.png")
