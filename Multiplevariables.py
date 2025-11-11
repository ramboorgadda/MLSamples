import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
df=pd.read_csv('homepricesMultiple.csv')
print(df.head())
sns.scatterplot(x='area',y='price',data=df)
plt.scatter(df.area, df.price, color='red', marker='o', linestyle='dashed')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()
sns.boxplot(x='bedrooms', data=df)
plt.xlabel('Bedrooms')
plt.title('Boxplot of Bedrooms')
plt.show()
median=df['bedrooms'].median()
print(median)
df['bedrooms']=df['bedrooms'].fillna(median)
print(df.head())
lin_reg=LinearRegression()
lin_reg.fit(df.drop('price', axis=1), df['price'])
print("Coefficients:", lin_reg.coef_)
print("Intercept:", lin_reg.intercept_)
print("Predicted price for area 5000,3 bedrooms and 10 years of Age:")
print(lin_reg.predict([[3000, 3, 10]]))  # Predicting price for area=5000, bedrooms=3, age=10