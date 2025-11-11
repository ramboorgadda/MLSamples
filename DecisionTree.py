import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
iris= load_iris()
print(iris.feature_names)
print(iris.target_names)
print(iris.data.shape)
X=pd.DataFrame(iris.data, columns=iris.feature_names)
y=pd.Series(iris.target, name='target')
print(X.head())
print(y.value_counts())
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
DecisionTreeClassifier_model = DecisionTreeClassifier()
DecisionTreeClassifier_model.fit(X_train, y_train)
y_pred = DecisionTreeClassifier_model.predict(X_test)
##visualize the decision tree
from sklearn.tree import plot_tree
plt.figure(figsize=(15, 10))
plot_tree(DecisionTreeClassifier_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title('Decision Tree Visualization')
plt.show()
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print(f'Predicted labels: {y_pred}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
# Prepruning anf Hyperparameter Tuning  
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [1, 2, 3, 4, 5],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')
grid_y_pred = grid_search.predict(X_test)
print(f'Accuracy after Grid Search: {accuracy_score(y_test, grid_y_pred)}')
print(f'Confusion Matrix after Grid Search:\n{confusion_matrix(y_test, grid_y_pred)}')
print(f'Classification Report after Grid Search:\n{classification_report(y_test, grid_y_pred)}')