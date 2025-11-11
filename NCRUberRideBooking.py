import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
dataframe=pd.read_csv("ncr_ride_bookings.csv")
print(dataframe.head())
print(dataframe.info())
# data cleaning
dataframe['Datetime']=pd.to_datetime(dataframe['Date']+' '+dataframe['Time'])
dataframe.drop(['Date','Time'],axis=1,inplace=True)
print(dataframe.head())
print(dataframe['Booking Status'].unique())
dataframe=dataframe[dataframe['Booking Status']!='No Driver Found']
print(dataframe['Booking Status'].unique())
dataframe['isCancelled']=dataframe['Booking Status'].apply(lambda x: 1 if 'Cancelled by Driver' in x else 0)
print(dataframe['isCancelled'].value_counts(normalize=True) * 100)
#categorical feature
cat_features=[feature for feature in dataframe.columns if dataframe[feature].dtype=='object'] 
for feature in cat_features:
    top_10_values=dataframe[feature].value_counts().nlargest(10)
    unique_values=dataframe[feature].nunique()
    print(f"unique values for categorical feature: {feature} is \n : {unique_values}")
    print(f"top ten values :{top_10_values}")
    plt.figure(figsize=(8,4))
    sns.countplot(data=dataframe[dataframe[feature].isin(top_10_values.index)],x=feature,hue='isCancelled',order=top_10_values.index)
    plt.title(f"Count plt for the top10 bookings {feature}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    #plt.show()
#Construct heatmap 
numerical_columns=dataframe.select_dtypes('number')
print(f"numerical column names {numerical_columns}")
print(numerical_columns.isna().sum())
for feature in numerical_columns:
    print(f"Top 10 distinct values for the feature {feature} is {dataframe[feature].value_counts().head(10)}")
correlation_matrix=numerical_columns.corr()
plt.figure(figsize=(7,6))
sns.heatmap(correlation_matrix,annot=True,fmt=".2f",cmap="coolwarm",square=True)
plt.title("correlation matirx")
plt.tight_layout()
#plt.show()
for feature in cat_features:
    unique_values=dataframe[feature].nunique()
    print(f"{feature} feature unique values are: {unique_values}")
low_cardinality_cols=['Vehicle Type',
                'Reason for cancelling by Customer',
                'Driver Cancellation Reason',
                'Incomplete Rides Reason',
                'Payment Method']
high_cardinality_cols=['Pickup Location','Drop Location']
dataframe['Payment Method']=dataframe['Payment Method'].fillna('Unknown')
X=dataframe.drop(columns=['Booking ID','Customer ID', 'isCancelled'])
y=dataframe['isCancelled']
#Pipeline Preprocessing
num_cols=X.select_dtypes(include=['int64','float64']).columns
cat_cols=X.select_dtypes(include=['object']).columns
# pipeline creation
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,precision_score,recall_score,f1_score
num_pipeline=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value=0)),
        ('scalar',StandardScaler())])
low_card_pipeline=Pipeline([('imputer',SimpleImputer(strategy='constant',fill_value='None')),
        ('onehot',OneHotEncoder(handle_unknown='ignore'))])
high_card_pipeline=Pipeline([
        ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))])
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('low_cat', low_card_pipeline, low_cardinality_cols),
    ('high_cat', high_card_pipeline, high_cardinality_cols)
])
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2,random_state = 10)
model_pipeline = Pipeline([
    ('preprocessor',preprocessor),
    ('model',RandomForestClassifier())
])
model_pipeline.fit(X_train, y_train)
y_pred=model_pipeline.predict(X_test)
print(f"confusion matrix \n :",confusion_matrix(y_test,y_pred))
print(f"classification Report \n :",classification_report(y_test,y_pred))
print(f"Accuracy score \n :",accuracy_score(y_test,y_pred))
print(f"Recall score \n :",recall_score(y_test,y_pred))
print(f"F1 score \n :",f1_score(y_test,y_pred))