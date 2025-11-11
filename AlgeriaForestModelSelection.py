import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv('Algerian_forest_fires_cleaned_dataset.csv')
print(df.head())
print(df.info())
df.drop(columns=['day','month','year'], inplace=True)
print(df.columns)
print(df['Classes'].value_counts())
df['Classes']=np.where(df['Classes'].str.contains('not fire'), 0, 1)
print(df['Classes'].value_counts())
print(df.tail())
X=df.drop('FWI',axis=1)
y=df['FWI']
print(X.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train.corr())
plt.figure(figsize=(12, 8))
sns.heatmap(X_train.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Features')
plt.show()
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: 
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr_features=correlation(X_train,0.85)
print(f'Features with correlation greater than 0.85: {corr_features}')
X_train.drop(columns=corr_features, inplace=True)
X_test.drop(columns=corr_features, inplace=True)
print(f'X_train shape after dropping correlated features: {X_train.shape}')
print(f'X_test shape after dropping correlated features: {X_test.shape}')
# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f'Scaled X_train : {X_train_scaled[:5]}')
print(f'Scaled X_test : {X_test_scaled[:5]}')
plt.subplots(figsize=(12, 8))
plt.subplot(1, 2, 1)
sns.boxplot(data=df)
plt.title('Boxplot of Features Before Scaling')
plt.subplot(1, 2, 2)
sns.boxplot(data=X_train_scaled)
plt.title('Boxplot of Features After Scaling')
plt.tight_layout()
plt.show()
#Linear Regression Model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)
y_pred = regressor.predict(X_test_scaled)
print(f'Regressor Coefficients: {regressor.coef_}')
print(f'Regressor Intercept: {regressor.intercept_}')
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
print(f'Root Mean Squared Error: {rmse}')
score = r2_score(y_test, y_pred)
print(f'R-squared Score: {score}')
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
#Lasso Regression
from sklearn.linear_model import Lasso
lasso_regressor = Lasso(alpha=0.1)
lasso_regressor.fit(X_train_scaled, y_train)
y_pred_lasso = lasso_regressor.predict(X_test_scaled)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
print(f'Lasso Mean Squared Error: {mse_lasso}')
print(f'Lasso Mean Absolute Error: {mae_lasso}')
print(f'Lasso Root Mean Squared Error: {rmse_lasso}')
sns.scatterplot(x=y_test, y=y_pred_lasso)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Lasso Regression)')
plt.show()
#Cross-Validation
from sklearn.linear_model import LassoCV
lassocv=LassoCV(cv=5)
lassocv.fit(X_train_scaled, y_train)
print(f'Optimal Alpha from LassoCV: {lassocv.alpha_}')
y_pred=lassocv.predict(X_test_scaled)
plt.scatter(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
#Ridge Regression model
from sklearn.linear_model import Ridge
Ridge_regressor = Ridge()
Ridge_regressor.fit(X_train_scaled, y_train)
y_pred_ridge = Ridge_regressor.predict(X_test_scaled)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
score = r2_score(y_test, y_pred_ridge)
print(f'Ridge Mean Squared Error: {mse_ridge}')
print(f'Ridge R-squared Score: {score}')
sns.scatterplot(x=y_test, y=y_pred_ridge)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (Ridge Regression)')
plt.show()     
from sklearn.linear_model import RidgeCV
ridgecv=RidgeCV(cv=5)
ridgecv.fit(X_train_scaled,y_train)
y_pred=ridgecv.predict(X_test_scaled)
plt.scatter(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
elastic.fit(X_train_scaled,y_train)
y_pred=elastic.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values (ElasticNet Regression)')
plt.show()

#pickle the machine learning model,preprocessing objects
import pickle
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(ridgecv, open('ridge_model.pkl', 'wb'))
