import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
DF=pd.read_csv('cardekho_imputated.csv', index_col=0)
import warnings
warnings.filterwarnings('ignore')
print(DF.head())
print(DF.info())
# Data Cleaning
print(DF.isnull().sum())
#Drop unnecessary columns
DF.drop(columns=['car_name','brand'],inplace=True)
print(DF.head())
print(DF.model.unique())
#Get All different types of Features
num_features = [feature for feature in DF.columns if DF[feature].dtype != 'object']
print(f'Numerical Features: {len(num_features)}')
cat_features = [feature for feature in DF.columns if DF[feature].dtype == 'object']
print(f'categorical Features: {len(cat_features)}')
discrete_features = [feature for feature in num_features if len(DF[feature].unique()) <= 25]
print(f'Discrete Features: {len(discrete_features)}')
continuous_features = [feature for feature in num_features if feature not in discrete_features]
print(f'Continuous Features: {len(continuous_features)}')
# Train Test Split
from sklearn.model_selection import train_test_split
X = DF.drop('selling_price', axis=1)
y = DF['selling_price']
print(X.head())
# Feature Engineering and scaling
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
print(len(DF['model'].unique()))
print(DF['model'].value_counts())
le=LabelEncoder()
X['model']=le.fit_transform(DF['model'])
print(f'X after encoding:\n {X.head()}')
print(len(X['seller_type'].unique()),len(X['fuel_type'].unique()),len(X['transmission_type'].unique()))
#Create a ColumnTransformer for OneHotEncoding
num_features = X.select_dtypes(exclude=["object"]).columns
onehot_columns = ['seller_type', 'fuel_type', 'transmission_type']
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
numeric_transformer = StandardScaler()
oh_encoder = OneHotEncoder(drop='first')
preprocessor = ColumnTransformer([
    ("OneHotEncoder",oh_encoder, onehot_columns),
    ("StandardScaler", numeric_transformer, num_features)
],remainder='passthrough')
X= preprocessor.fit_transform(X)
print(pd.DataFrame(X).head())
X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.25, random_state=42)
#Model Training and Model Selection
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#Create a function to evaluate models
def evaluate_model(true, predicted):
    mse = mean_squared_error(true, predicted)
    mae = mean_absolute_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    return mae,rmse,r2
## Begining of Model Training
models = {
    'Linear Regression': LinearRegression(),
    'Lasso': Lasso(),
    'Ridge': Ridge(),
    'KNN Regressor': KNeighborsRegressor(),
    'Decision Tree Regressor': DecisionTreeRegressor(),
    'Random Forest': RandomForestRegressor(),
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train,y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    model_train_mae,model_train_rmse,model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae,model_test_rmse,model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    print(f'Model: {model}')
    print(f'Train MAE: {model_train_mae}, Train RMSE: {model_train_rmse}, Train R2: {model_train_r2}')
    print(f'Test MAE: {model_test_mae}, Test RMSE: {model_test_rmse}, Test R2: {model_test_r2}')
    print('-' * 50)
# Hyperparameter Tuning
knn_params = {
    'n_neighbors': [2, 3, 10, 20, 40, 50]}
rf_params = {
    'n_estimators': [100, 200, 500,1000],
    'max_depth': [5, 8, 15, None, 10],
    'max_features': [5, 7, "auto", 8],
    'min_samples_split': [2, 8, 15,20]}

randomcv_models=[('KNN Regressor', KNeighborsRegressor(), knn_params),
('Random Forest', RandomForestRegressor(), rf_params)]
#Hyperparameter Tuning
from sklearn.model_selection import RandomizedSearchCV
model_params={}
for model_name, model, params in randomcv_models:
    random_search = RandomizedSearchCV(model, params, n_iter=100, cv=3,verbose=2, n_jobs=-1)
    random_search.fit(X_train, y_train)
    model_params[model_name] = random_search.best_params_
    
for model_name in model_params:
    print(f"---------------- Best Params for {model_name} -------------------")
    print(model_params[model_name])
# Select the best model based on the hyperparameter tuning
models = {
    "KNN Regressor": KNeighborsRegressor(n_neighbors=10),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=15, max_features=5, min_samples_split=2)
}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    model_train_mae,model_train_rmse,model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae,model_test_rmse,model_test_r2 = evaluate_model(y_test, y_test_pred)
    
    print(f'Model: {model}')
    print(f'Train MAE: {model_train_mae}, Train RMSE: {model_train_rmse}, Train R2: {model_train_r2}')
    print(f'Test MAE: {model_test_mae}, Test RMSE: {model_test_rmse}, Test R2: {model_test_r2}')
    print('-' * 50)