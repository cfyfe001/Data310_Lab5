# Lab 5

## Question 1: In the case of  kernel Support Vector Machines for classification, such as the radial basis function kernel,  one or more landmark points are considered by the algorithm.
This is true! We want the classification methods to do this!


## Question 2: A hard margin SVM is appropriate for data which is not linearly separable.
This is false. According to our notes, we do not use hard margin in this scenario - it should be soft.


## Question 3: In K-nearest neighbors, all observations that fall within a circle with radius of K are included in the estimation for a new point.
This is false. This is not how K-nearest neighbors works - it uses the closest K number of points to determine which group it belongs to.


## Question 4: For the breast cancer data (from sklearn library), if you choose a test size of 0.25 (25% of your data), with a random_state of 1693, how many observations are in your training set?
```markdown
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)

X = df.values
y = dat.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1693)

X_train.shape
```
As a result, it is evident that there are 426 observations in the training set.


## Question 5: Kernel SVM is only applicable if you have at least 3 independent variables (3 dimensions).
This is false. You need at least 2 independent variables.


## Question 6: Using your Kernel SVM model with a radial basis function kernel, predict the classification of a tumor if it has a radius mean of 16.78 and a texture mean of 17.89.
```markdown
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

svc_model = SVC()
svc_model.fit(X, y)

svc_model.predict([[16.78,17.89]])
```
As a result, the answer is [0], so it is malignant. 


## Question 7: Using your logistic model, predict the probability a tumor is malignant if it has a radius mean of 15.78 and a texture mean of 17.89.
```markdown
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

model = LogisticRegression()
model.fit(X,y)
p = model.predict_proba([[15.78,17.89]])
p[0:,]
```
Based on this code, we understand that the probability is about 67.2%. Since this was not an answer choice, I selected the closest one - 61%.


## Question 8: Using your nearest neighbor classifier with k=5 and weights='uniform', predict if a tumor is benign or malignant if the Radius Mean  is 17.18, and the Texture Mean is 8.65
```markdown
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from sklearn.neighbors import KNeighborsClassifier

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

model = KNeighborsClassifier(n_neighbors=5,weights="uniform")

model.fit(X,y)
model.predict([[17.18,8.65]])
```
The code tells us that the answer is [1], so it is benign.


## Question 9: Consider a RandomForest classifier with 100 trees, max depth of 5 and random state 1234. From the data consider only the "mean radius" and the "mean texture" as the input features. If you apply a 10-fold stratified cross-validation and estimate the mean AUC (based on the receiver operator characteristics curve) the answer is
```markdown
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix as CM
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
from sklearn.neighbors import KNeighborsClassifier

dat = load_breast_cancer()
df = pd.DataFrame(data=dat.data, columns=dat.feature_names)
df = df.drop(columns=['mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error','concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension'])
X = df.values
y = dat.target

cv = StratifiedKFold(n_splits=10)
classifier = RandomForestClassifier(n_estimators=100,max_depth=5,random_state=1234)


tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))
for i, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = plot_roc_curve(classifier, X[test], y[test],
                         name='ROC fold {}'.format(i),
                         alpha=0.3, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b',
        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
       title="Receiver operating characteristic example")
ax.legend(loc="lower right")
plt.show()

print(mean_auc)
```
We understand that the answer is 0.9494 as a result of the code.


## Question 10: What is one reason simple linear regression (OLS) is not well suited to calculating the probability of discrete cases?
The main issue with OLS concerning discrete probabilities is that it can result in predicted probabilities greater than 1 or less than 0, so it does not act well in some circumstances.


## Question 11: When applying the K - Nearest Neighbors classifier we always get better results if the weights are changed from 'uniform' to 'distance'.
This is false. As with most things in machine learning, each data set reacts differently to different parameters. We can never assume this statement to be true.
