import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
df_index=pd.read_csv('economic_index.csv')
print(df_index.head())
df_index.drop(columns=['Unnamed: 0','year','month'], inplace=True)
print(df_index.head())
sns.pairplot(df_index)
plt.show()
print(df_index.isnull().sum())
print(df_index.corr())
X=df_index.iloc[:,:-1]
y=df_index.iloc[:,-1]
print(X.shape, y.shape)
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=42)
#sns.regplot(df_index['interest_rate'],df_index['index_price'],data=df_index)
#plt.show()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
print(f'Shape of X_train: {X_train.shape}, y_train: {y_train.shape}')
print(X_train)
print(X_test)
print(y_train)
print(y_test)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
## cross validation
from sklearn.model_selection import cross_val_score
validation_score=cross_val_score(regressor,X_train,y_train,scoring='neg_mean_squared_error',
                                cv=3)
print(f'Cross Validation Score: {validation_score.mean()}')
print(f'Regressor Coefficients: {regressor.coef_}')
print(f'Regressor Intercept: {regressor.intercept_}')
y_pred= regressor.predict(X_test)
print(f'Predicted Values: {y_pred}')
## Performance Metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print(score)
#display adjusted R-squared
print(f' X test shape is: {X_test.shape[1]}')
print(1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1))

sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.show()
residuals= y_test - y_pred
sns.displot(residuals, kind="kde", fill=True)
plt.xlabel('Residuals')
plt.title('Residuals Distribution')
plt.show()
import statsmodels.api as sm
model=sm.OLS(y_train,X_train).fit()
print(model.summary())