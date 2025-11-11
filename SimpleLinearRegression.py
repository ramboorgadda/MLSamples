import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
dataframe=pd.read_csv('homeprices.csv')
print(dataframe.head())
LR=LinearRegression()
LR.fit(dataframe[['area']],dataframe.price)
print(LR.coef_)
print(LR.intercept_)
sns.scatterplot(x='area',y='price',data=dataframe)
plt.plot(dataframe.area,dataframe.price,color='red',marker='o',linestyle='dashed')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Area vs Price')
plt.show()
print(LR.predict([[5000]]))  # Predicting price for an area of 5000
df=pd.read_csv('areas.csv')
print(df.head())
Price=LR.predict(df)  # Predicting prices for the areas in areas.csv
df['price']=Price
print(df.head())
df.to_csv('predicted_prices.csv',index=False)  # Saving the predictions to a new CSV file
print("Predictions saved to 'predicted_prices.csv'")