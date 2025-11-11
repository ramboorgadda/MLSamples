import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
DF=pd.read_csv('loan_approval_dataset.csv')
print(DF.head())
print(DF.info())
print(DF.isnull().sum())
print(DF.describe())
DF.drop('loan_id',axis=1,inplace=True)
# Data Cleaning

feature_categories =[feature for feature in DF.columns if DF[feature].dtype == 'object']
print(f'length of categorical features: {len(feature_categories)}')
for feature in feature_categories:
    print(f'unique values are {DF[feature].unique()}')
feature_numerical=[feature for feature in DF.columns if DF[feature].dtype !='object']
print(f'length of numerical features: {len(feature_numerical)}')
for feature in feature_numerical:
    print(f'unique values are {DF[feature].unique()}')
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,f1_score
num_col=DF.select_dtypes(include=['int','float'])
print(num_col)
cat_col=DF.select_dtypes(include=['object'])
label=LabelEncoder()
for col in cat_col:
    DF[col]=label.fit_transform(DF[col])
print(DF.head())
X=DF.drop('loan_status',axis = 1)
y=DF['loan_status']
X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=0.8,random_state=42)
scaler= StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
logisticsRegression=LogisticRegression(random_state=42,max_iter=1000)
logisticsRegression.fit(X_train,y_train)
y_pred=logisticsRegression.predict(X_test)
precision_score=precision_score(y_test,y_pred)
recall_score=recall_score(y_test,y_pred)
f1_score=f1_score(y_test,y_pred)
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred))
print("Classification Report:\n",classification_report(y_test,y_pred))
print("Precession score",precision_score)
print("recall score",recall_score)
print("F1 score",f1_score)
decessiontree=DecisionTreeClassifier(random_state=42)
decessiontree.fit(X_train,y_train)
y_pred=decessiontree.predict(X_test)
print("Confusion Matrix for decession Tree: \n",confusion_matrix(y_test,y_pred))
print("Classification Matrix for decession Tree: \n",classification_report(y_test,y_pred))
precession_score=precision_score(y_test,y_pred)
recall_Score=recall_score(y_test,y_pred)
f1_score=f1_score(y_test,y_pred)

print("Precession for decession tree :",precision_score)
print("Recall for decession tree :",recall_Score)
print("f1 score for decession tree:",f1_score)

