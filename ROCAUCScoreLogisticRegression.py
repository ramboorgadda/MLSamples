import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
X, y = make_classification(n_samples=1000, n_classes=2, random_state=1)
print(X.shape)
print(y.shape)
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
def evaluvate_model(model, X_test, y_test):
    model_prob = model.predict_proba(X_test)[:, 1]
    auc_score = roc_auc_score(y_test, model_prob)
    print(f'Model AUC Score: {auc_score}')
    return auc_score
auc_score=evaluvate_model(model, X_test, y_test)
print(f' auc score is  : {auc_score}')

## calculate ROC Curves
def estimate_roccurve(model, X_test, y_test):
    model_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, model_prob)
    return fpr, tpr, thresholds
dummy_fpr, dummy_tpr, _ = estimate_roccurve(model,X_test, y_test)
model_fpr, model_tpr, thresholds = estimate_roccurve(model, X_test, y_test)


print(f'Dummy Model FPR: {dummy_fpr[:5]}, TPR: {dummy_tpr[:5]}')
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