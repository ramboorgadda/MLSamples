import numpy as np
import seaborn as sns
import matplotlib.pyplot as plot
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
X = 6 * np.random.rand(100, 1) - 3
print(f' value of  {X}')

y =0.5 * X**2 + 1.5*X + 2 + np.random.randn(100, 1)
print(f'y value {y}')
sns.scatterplot(x=X[:, 0], y=y[:, 0])
plot.xlabel('X')
plot.ylabel('y')
plot.title('Scatter Plot of X vs y')
plot.show()
# split the data into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)
Regression=LinearRegression()
Regression.fit(X_train,y_train)
print('Coefficients:',Regression.coef_)
print('Intercept:',Regression.intercept_)
from sklearn.metrics import  r2_score
r2_score_value = r2_score(y_test, Regression.predict(X_test))
print(f'R^2 Score: {r2_score_value}')
plot.plot(X_train, Regression.predict(X_train), color='red', label='Regression Line')
plot.scatter(X_train,y_train, color='blue', label='Training Data')
plot.xlabel('X DataSet')
plot.ylabel('y DataSet')
plot.show()
# Apply polynomial Regression
poly_features = PolynomialFeatures(degree=2,include_bias=True)
X_train_poly=poly_features.fit_transform(X_train)
X_test_poly=poly_features.transform(X_test)
print(f'X_train_poly :{X_train_poly}')
print(f'X_test_poly :{X_test_poly}')
print(X_train_poly.shape, X_test_poly.shape)
from sklearn.metrics import r2_score
regression = LinearRegression()
regression.fit(X_train_poly, y_train)
y_pred = regression.predict(X_test_poly)
score=r2_score(y_test,y_pred)
print(f'R2 Score after polynomial:{score}')
print(f'Coefficients after polynomial:{regression.coef_}')
print(f'Intercept after polynomial:{regression.intercept_}')
plot.scatter(X_train,regression.predict(X_train_poly), color='blue', label='Training Data')
plot.scatter(X_train,y_train)
plot.show()

poly=PolynomialFeatures(degree=3,include_bias=True)
X_train_poly=poly.fit_transform(X_train)
X_test_poly=poly.transform(X_test)
print(f'X_train_poly :{X_train_poly}')
print(f'X_test_poly :{X_test_poly}')
#3 Prediction of new data set
X_new = np.linspace(-3, 3, 200).reshape(200, 1)
X_new_poly = poly.transform(X_new)
print(X_new_poly)
y_new=regression.predict(X_new_poly)
plot.plot(X_new,y_new, "r-", label='Polynomial Regression Line', linewidth=2)
plot.scatter(X_train, y_train, color='blue', label='Training Data')
plot.scatter(X_test, y_test, color='green', label='Test Data')
plot.xlabel('X DataSet')
plot.ylabel('y DataSet')
plot.title('Polynomial Regression')
plot.legend()
plot.show()