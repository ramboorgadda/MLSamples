import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
DF=pd.read_csv('loan_approval_dataset.csv')
print(DF.head())
print(DF.info())
print(DF.isnull().sum())
# Data Cleaning

