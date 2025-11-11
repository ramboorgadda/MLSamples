from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
load_iris_data = load_iris()
X = load_iris_data.data
y = load_iris_data.target
print(f'Feature Names: {load_iris_data.feature_names}')
print(f'Target Names: {load_iris_data.target_names}')
print(f'X shape: {X.shape}, y shape: {y.shape}')
print(f'First 5 rows of X:\n{X[:5]}')
print(f'First 5 rows of y: {y[:5]}')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f'Train shape: {X_train.shape}, Test shape: {X_test.shape}')
print(f'Train labels shape: {y_train.shape}, Test labels shape: {y_test.shape}')
#histogram of features
plt.figure(figsize=(12, 6))
for i in range(X.shape[1]):
    plt.subplot(2, 2, i + 1)
    sns.histplot(X[:, i], kde=True, bins=30)
    plt.title(load_iris_data.feature_names[i])
plt.tight_layout()
plt.show()
GaussianNB_model = GaussianNB()
GaussianNB_model.fit(X_train, y_train)
y_pred = GaussianNB_model.predict(X_test)
print(f'Predicted labels: {y_pred}')
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')