import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
from sklearn.linear_model import LinearRegression
from word2number import wtn
dataframe=pd.read_csv('hiring.csv')
print(dataframe.head())
dataframe['experience'].fillna('zero', inplace=True)  # Filling NaN values in 'experience' with 0
print(dataframe.head())
dataframe['experience'] =  # Converting experience to numeric values
print(dataframe.head())