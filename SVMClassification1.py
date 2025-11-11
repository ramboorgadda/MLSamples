import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#Lets Create Synthetic Data
from sklearn.datasets import make_classification
X,y=make_classification(n_samples=1000, n_features=10,n_classes=2,n_clusters_per_class=2,n_redundant=0)
print(X)
print(y)
print(pd.DataFrame(X[0]))
sns.scatterplot(x=X[:,0], y=X[:,1], hue=y, palette='coolwarm')
plt.title('Synthetic Data Scatter Plot')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}, Test labels shape: {y_test.shape}')
from sklearn.svm import SVC
svc=SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)
print(f'Predicted labels: {y_pred}')
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
rbf=SVC(kernel='rbf')
rbf.fit(X_train, y_train)
y_pred_rbf=rbf.predict(X_test)
print(f'RBF Kernel Predicted labels: {y_pred_rbf}')
print(f'RBF Kernel Accuracy: {accuracy_score(y_test, y_pred_rbf)}')
print(f'RBF Kernel Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rbf)}')
print(f'RBF Kernel Classification Report:\n{classification_report(y_test, y_pred_rbf)}')
polynomial=SVC(kernel='poly')
polynomial.fit(X_train,y_train)
## Prediction
y_pred2=polynomial.predict(X_test)
print(f'Polynomial Kernel Predicted labels: {y_pred2}')
print(f'Polynomial Kernel Accuracy: {accuracy_score(y_test, y_pred2)}')
print(f'Polynomial Kernel Confusion Matrix:\n{confusion_matrix(y_test, y_pred2)}')
print(f'Polynomial Kernel Classification Report:\n{classification_report(y_test, y_pred2)}')
sigmoid=SVC(kernel='sigmoid')
sigmoid.fit(X_train,y_train)
## Prediction
y_pred3=sigmoid.predict(X_test)
print(f'Sigmoid Kernel Predicted labels: {y_pred3}')
print(f'Sigmoid Kernel Accuracy: {accuracy_score(y_test, y_pred3)}')
print(f'Sigmoid Kernel Confusion Matrix:\n{confusion_matrix(y_test, y_pred3)}')
print(f'Sigmoid Kernel Classification Report:\n{classification_report(y_test, y_pred3)}')