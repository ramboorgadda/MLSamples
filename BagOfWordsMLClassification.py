import numpy as np
import pandas as pd
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
messages=pd.read_csv('SMSSpamCollection.txt',sep='\t',names=['label','message'])
print(messages.head())
corpus=[]
ps=PorterStemmer()
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    review=[ps.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
print(f'corpus {corpus}')
#Create Bag of Words
y=pd.get_dummies(messages['label'])
print(f"{y}")
y=y.iloc[:,0].values
print(f"iloc values {y.unique}")
# train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(corpus,y,test_size=.20)
#create bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2500,ngram_range=(1,2))
print(len(X_train),len(y_train))
X_train=cv.fit_transform(X_train).toarray()
X_test=cv.transform(X_test).toarray()
print(cv.vocabulary_)
print(f"train data {X_train.reshape}")
print(f"test data {X_test.reshape}")
from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_pred=mnb.predict(X_test)
from sklearn.metrics import classification_report,accuracy_score
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))