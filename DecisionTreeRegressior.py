from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
# Load the diabetes dataset
load_diabetes_data = load_diabetes()
print(load_diabetes_data['DESCR'])
df_diabetes=pd.DataFrame(load_diabetes_data.data, columns=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6'])
print(df_diabetes.head())
X=df_diabetes
y=load_diabetes_data['target']
print(f'Feature Names: {load_diabetes_data.feature_names}')
print(f'Target Name: {y}')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
print(X_train.corr())
plt.figure(figsize=(10, 8))
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()
# Initialize the Decision Tree Regressor
decision_tree_regressor = DecisionTreeRegressor()
decision_tree_regressor.fit(X_train, y_train)
#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
param_grid = {
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    'max_depth': [2, 4, 6, 8, 10],
    'splitter': ['best', 'random'],
    'max_features': ['auto', 'sqrt', 'log2']
}
grid_search = GridSearchCV(estimator=decision_tree_regressor, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)
import warnings
warnings.filterwarnings("ignore")
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_}')
grid_y_pred = grid_search.predict(X_test)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, grid_y_pred)
mae = mean_absolute_error(y_test, grid_y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
score = r2_score(y_test, grid_y_pred)
print(f'R-squared Score: {score}')
selectedmodel=DecisionTreeRegressor(criterion='friedman_mse',max_depth=4,max_features='log2',splitter='random')
selectedmodel.fit(X_train, y_train)
y_pred = selectedmodel.predict(X_test)
print(f'Predicted Values: {y_pred}')
print(f'Best Parameters: {selectedmodel.get_params()}')

# Performance Metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
score = r2_score(y_test, y_pred)
print(f'R-squared Score: {score}')
# Visualize the predictions
plt.figure(figsize=(10, 6))
from sklearn import tree
tree.plot_tree(selectedmodel, filled=True)
plt.show()
