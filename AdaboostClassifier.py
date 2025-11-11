import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
DataFrame=pd.read_csv('Travel.csv')
print(DataFrame.head())
print(DataFrame.info())
# Data Cleaning
print(DataFrame.isnull().sum())
print(DataFrame.Gender.value_counts())
print(DataFrame.Occupation.value_counts())
print(DataFrame.TypeofContact.value_counts())
print(DataFrame.MaritalStatus.value_counts())
DataFrame['Gender']=DataFrame['Gender'].replace('Fe Male','Female')
DataFrame['MaritalStatus']=DataFrame['MaritalStatus'].replace('Unmarried','Single')
print(DataFrame['Gender'].value_counts())
print(DataFrame['MaritalStatus'].value_counts())
# Checking missing values
feature_with_na=[feature for feature in DataFrame.columns if DataFrame[feature].isnull().sum()>=1]
for feature in feature_with_na:
    print(feature,'has' ,np.round(DataFrame[feature].isnull().mean()*100,5), 'missing values')
# statistics of numerical features
print(DataFrame[feature_with_na].select_dtypes(exclude='object').describe())
# Filling missing values
# print NumberOfFollowups value counts

DataFrame.Age.fillna(DataFrame.Age.median(), inplace=True)
DataFrame.TypeofContact.fillna(DataFrame.TypeofContact.mode()[0], inplace=True)
DataFrame.NumberOfFollowups.fillna(DataFrame.NumberOfFollowups.mode()[0], inplace=True)
DataFrame.DurationOfPitch.fillna(DataFrame.DurationOfPitch.median(), inplace=True)
DataFrame.PreferredPropertyStar.fillna(DataFrame.PreferredPropertyStar.mode()[0], inplace=True)
DataFrame.NumberOfTrips.fillna(DataFrame.NumberOfTrips.mode()[0], inplace=True)
DataFrame.NumberOfChildrenVisiting.fillna(DataFrame.NumberOfChildrenVisiting.mode()[0], inplace=True)
DataFrame.NumberOfFollowups.fillna(DataFrame.NumberOfFollowups.mode()[0], inplace=True)
DataFrame.MonthlyIncome.fillna(DataFrame.MonthlyIncome.median(), inplace=True)

print(DataFrame.isnull().sum())
print(DataFrame.head())
DataFrame['TotalVisiting']= DataFrame['NumberOfChildrenVisiting'] + DataFrame['NumberOfPersonVisiting']
DataFrame.drop(['CustomerID','NumberOfChildrenVisiting', 'NumberOfPersonVisiting'], axis=1, inplace=True)
print(DataFrame.head())
print(DataFrame['TotalVisiting'].value_counts())
#Feature Engineering
num_features= [feature for feature in DataFrame.columns if DataFrame[feature].dtype != 'object']
print(f'Numerical Features: {len(num_features)}')
cat_features= [feature for feature in DataFrame.columns if DataFrame[feature].dtype == 'object']
print(f'Categorical Features: {len(cat_features)}')
discrete_features= [feature for feature in num_features if len(DataFrame[feature].unique()) <= 25]
print(f'Discrete Features: {len(discrete_features)}')
continuous_features= [feature for feature in num_features if feature not in discrete_features]
print(f'Continuous Features: {len(continuous_features)}')
# Train Test Split
from sklearn.model_selection import train_test_split
X = DataFrame.drop('ProdTaken', axis=1)
y = DataFrame['ProdTaken']
print(X.head())
print(y.value_counts())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(X.info())
# Create Column Transformer with 3 types of transformers
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# Create Column Transformer with 3 types of transformers
cat_features=X.select_dtypes(include='object').columns
num_features=X.select_dtypes(exclude='object').columns
Oh_transformer = OneHotEncoder(drop='first', sparse_output=False)
numeric_transformer = StandardScaler()
preprocessor = ColumnTransformer( [
    ('categorical', Oh_transformer, cat_features),
    ('numerical', numeric_transformer, num_features)
])
print(preprocessor)
X_train=preprocessor.fit_transform(X_train)
X_test=preprocessor.transform(X_test)
#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,precision_score,recall_score,f1_score
# Initialize models
models={
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Ada boost': AdaBoostClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression()
}
for i in range(len(list(models))):
    
    model=list(models.values())[i]
    model.fit(X_train, y_train)
    y_trainpred=model.predict(X_train)
    y_testpred=model.predict(X_test)
    
    print(f'Accuracy of {model}: {accuracy_score(y_train, y_trainpred)}')
    print(f'Confusion Matrix of {model}:\n{confusion_matrix(y_train, y_trainpred)}')
    print(f'Classification Report of {model}:\n{classification_report(y_train, y_trainpred)}')
    print(f'Precision of {model}: {precision_score(y_train, y_trainpred)}')
    print(f'Recall of {model}: {recall_score(y_train, y_trainpred)}')
    print(f'F1 Score of {model}: {f1_score(y_train, y_trainpred)}\n')
    
    print(f'Accuracy of {model}: {accuracy_score(y_test, y_testpred)}')
    print(f'Confusion Matrix of {model}:\n{confusion_matrix(y_test, y_testpred)}')
    print(f'Classification Report of {model}:\n{classification_report(y_test, y_testpred)}')
    print(f'Precision of {model}: {precision_score(y_test, y_testpred)}')
    print(f'Recall of {model}: {recall_score(y_test, y_testpred)}')
    print(f'F1 Score of {model}: {f1_score(y_test, y_testpred)}\n')
    print(f'{"-"*50}\n')
    #Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_curve, auc
rf_params = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [10, 20, 30, 40, 50],
    'min_samples_split': [2,8,15,20],
    'max_features': [5,7,'auto',8]
}
adaboost_param={
    "n_estimators":[50,60,70,80,90],
    "algorithm":['SAMME','SAMME.R']
} 
random_model =[
                ('RF',RandomForestClassifier(),rf_params),
                ('AdaBoost', AdaBoostClassifier(), adaboost_param)]
print(random_model)
model_params={}
for name, model, params in random_model:
    print(f'Tuning {name}...')
    random_search = RandomizedSearchCV(model, params, n_iter=100, cv=5, scoring='accuracy', n_jobs=-1, random_state=42)
    random_search.fit(X_train, y_train)
    model_params[name] = random_search.best_params_
for model_name in model_params:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_params[model_name])
    
models={
    "Adaboost": AdaBoostClassifier(n_estimators=50, algorithm='SAMME')
}
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train, y_train)
    y_trainpred=model.predict(X_train)
    y_testpred=model.predict(X_test)

    
    print(f'Accuracy of {model}: {accuracy_score(y_train, y_trainpred)}')
    print(f'Confusion Matrix of {model}:\n{confusion_matrix(y_train, y_trainpred)}')
    print(f'Classification Report of {model}:\n{classification_report(y_train, y_trainpred)}')
    print(f'Precision of {model}: {precision_score(y_train, y_trainpred)}')
    print(f'Recall of {model}: {recall_score(y_train, y_trainpred)}')
    print(f'F1 Score of {model}: {f1_score(y_train, y_trainpred)}\n')
    from sklearn.metrics import roc_curve, roc_auc_score
    print(f'Roc AUC Score of {model}: {roc_auc_score(y_train, y_trainpred)}')
    print(f'{"-"*50}\n')
    
    print(f'Accuracy of {model}: {accuracy_score(y_test, y_testpred)}')
    print(f'Confusion Matrix of {model}:\n{confusion_matrix(y_test, y_testpred)}')
    print(f'Classification Report of {model}:\n{classification_report(y_test, y_testpred)}')
    print(f'Precision of {model}: {precision_score(y_test, y_testpred)}')
    print(f'Recall of {model}: {recall_score(y_test, y_testpred)}')
    print(f'F1 Score of {model}: {f1_score(y_test, y_testpred)}\n')
    print(f'Roc AUC Score of {model}: {roc_auc_score(y_test, y_testpred)}')
    print(f'{"-"*50}\n')
    
from sklearn.metrics import roc_auc_score,roc_curve
plt.figure()

# Add the models to the list that you want to view on the ROC plot
auc_models = [
{
    'label': 'AdaBoost Classifier',
    'model': AdaBoostClassifier(n_estimators=70,algorithm='SAMME'),
    'auc':  0.6049
},
    
]
# create loop through all model
for algo in auc_models:
    model = algo['model'] # select the model
    model.fit(X_train, y_train) # train the model
# Compute False postive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
# Calculate Area under the curve to display on the plot
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))
# Custom settings for the plot 
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show() 