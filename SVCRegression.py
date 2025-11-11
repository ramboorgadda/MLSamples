import seaborn as sns
import numpy as np
import pandas as pd
df=sns.load_dataset('tips')
print(df.head())
print(df.info())
print(df['sex'].value_counts())
print(df['day'].value_counts())
print(df['time'].value_counts())
print(df['smoker'].value_counts())
print(df.describe())
#Feature encoding
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

#Independent and dependent features
print(df.columns)
X=df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
y=df['total_bill']
print(X.head())
print(y.head())
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)
print(f'X_train shape: {X_train.head()}')
print(f'X_test shape: {X_test.head()}')
print(f'y_train shape: {y_train.head()}')
print(f'y_test shape: {y_test.head()}')
from sklearn.preprocessing import LabelEncoder
le1=LabelEncoder()
le2=LabelEncoder()
le3=LabelEncoder()
X_train['sex']=le1.fit_transform(X_train['sex'])
X_train['time']=le2.fit_transform(X_train['time'])
X_train['smoker']=le3.fit_transform(X_train['smoker'])
print(f'X_train after encoding:\n {X_train.head()}')
X_test['sex']=le1.transform(X_test['sex'])
X_test['time']=le2.transform(X_test['time'])
X_test['smoker']=le3.transform(X_test['smoker'])
print(f'X_test after encoding:\n {X_test.head()}')
#one hot encoding 
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct=ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), ['day'])], remainder='passthrough')
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
X_train=ct.fit_transform(X_train)
X_test=ct.transform(X_test)
print(f'X_train after one hot encoding:\n {X_train[:5]}')
print(f'X_test after one hot encoding:\n {X_test[:5]}')
## SVR--Support Vector Regression
from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
print(f'Predicted values: {y_pred[:5]}')
print(f'Actual values: {y_test[:5].values}')
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
print(f'Mean Squared Error: {mean_squared_error(y_test, y_pred)}')
print(f'Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}')
print(f'R-squared Score: {r2_score(y_test, y_pred)}')