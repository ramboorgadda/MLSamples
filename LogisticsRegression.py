import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

# Create the dataset
X,y=make_classification(n_samples=1000, n_features=10,n_classes=2, random_state=42)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
print(f'X_train shape: {X_train.shape}')
print(f'X_test shape: {X_test.shape}')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
logistic= LogisticRegression()
logistic.fit(X_train, y_train)
y_pred=logistic.predict(X_test)
print(y_pred)
print(f'Accuracy: {accuracy_score(y_test,y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
#Hyper Parameter Tuning and Cross validation
from sklearn.model_selection import GridSearchCV
model=LogisticRegression()
penaltys = ['l1', 'l2', 'elasticnet']
c_vals = [0.01, 0.1, 1, 10, 100]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
params=dict(penalty=penaltys, C=c_vals, solver=solver)
from sklearn.model_selection import StratifiedKFold
cv=StratifiedKFold()
# Grid Search
from sklearn.model_selection import GridSearchCV
grid= GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1)
print(f'Grid is : {grid}')
grid.fit(X_train, y_train)
print(f'Best Parameters: {grid.best_params_}')
print(f'Best Score: {grid.best_score_}')
y_pred=grid.predict(X_test)
print(f'Accuracy after Grid Search: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix after Grid Search:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report after Grid Search:\n{classification_report(y_test, y_pred)}')
#Randomized Search
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
model=LogisticRegression()
randomcv=RandomizedSearchCV(estimator=model,param_distributions=params,cv=5,scoring='accuracy')
randomcv.fit(X_train, y_train)
print(f'Best Parameters after Randomized Search: {randomcv.best_params_}')
print(f'Best Score after Randomized Search: {randomcv.best_score_}')
y_pred=randomcv.predict(X_test)
print(f'Accuracy after Randomized Search: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix after Randomized Search:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report after Randomized Search:\n{classification_report(y_test, y_pred)}')
#Logistic Regression For Multiclass Classification Problem
X, y = make_classification(n_samples=1000, n_features=10,n_informative=3, n_classes=3, random_state=15)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
logistic = LogisticRegression(multi_class='ovr', solver='lbfgs')
logistic.fit(X_train, y_train)
logistic_pred = logistic.predict(X_test)
print(f'Accuracy for Multiclass Classification: {accuracy_score(y_test, logistic_pred)}')
print(f'Confusion Matrix for Multiclass Classification:\n{confusion_matrix(y_test, logistic_pred)}')
print(f'Classification Report for Multiclass Classification:\n{classification_report(y_test, logistic_pred)}')
#Logistic Regression with Imbalanced Classes
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=10000, n_features=2, n_clusters_per_class=1,n_redundant=0, weights=[0.99], random_state=42)
print(X.shape, y.shape)
print(f'Number of samples: {len(y)}')
print(f'Number of classes: {len(np.unique(y))}')
print(f'Class distribution: {np.bincount(y)}')

sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='coolwarm')
plt.title('Imbalanced Classes')
plt.show()
#Hyperparameter Tuning for Imbalanced Classes
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV
model=LogisticRegression()
penaltys = ['l1', 'l2', 'elasticnet']
c_vals = [0.01, 0.1, 1, 10, 100]
solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
class_weight=[{0:w,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]
print(f'Class Weights: {class_weight}')
params=dict(penalty=penaltys, C=c_vals, solver=solver, class_weight=class_weight)
print(f'Parameters for Imbalanced Classes: {params}')
cv=StratifiedKFold()
grid=GridSearchCV(estimator=model, param_grid=params, cv=cv, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)
print(f'Best Parameters for Imbalanced Classes: {grid.best_params_}')
print(f'Best Score for Imbalanced Classes: {grid.best_score_}')
print(f'Accuracy for Imbalanced Classes: {grid.score(X_test, y_test)}')
y_pred=grid.predict(X_test)
print(f'Confusion Matrix for Imbalanced Classes:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report for Imbalanced Classes:\n{classification_report(y_test, y_pred)}')
#Logistic Regression With ROC curve And ROC AUC score

X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
dummy_model_prob= [0 for _ in range(len(y_test))]
print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'Dummy Model Probabilities: {dummy_model_prob[:5]}')
model = LogisticRegression()
model.fit(X_train, y_train)
model_prob= model.predict_proba(X_test)
print(model_prob)
print(f'Model Probabilities: {model_prob[:,1]}')
## Lets calulcate the scores
dummy_model_auc=roc_auc_score(y_test,dummy_model_prob)
model_auc=roc_auc_score(y_test,model_prob)
print(dummy_model_auc)
print(model_auc)

## calculate ROC Curves
dummy_fpr, dummy_tpr, _ = roc_curve(y_test, dummy_model_prob)
model_fpr, model_tpr, thresholds = roc_curve(y_test, model_prob)
# plot the roc curve for the model
plt.plot(dummy_fpr, dummy_tpr, linestyle='--', label='Dummy Model')
plt.plot(model_fpr, model_tpr, marker='.', label='Logistic')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()

fig = plt.figure(figsize=(20,50))
plt.plot(dummy_fpr, dummy_tpr, linestyle='--', label='Dummy Model')
plt.plot(model_fpr, model_tpr, marker='.', label='Logistic')
ax = fig.add_subplot(111)
for xyz in zip(model_fpr, model_tpr,thresholds):   
    ax.annotate('%s' % np.round(xyz[2],2), xy=(xyz[0],xyz[1]))
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
# show the legend
plt.legend()
# show the plot
plt.show()