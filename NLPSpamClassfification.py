import pandas as pd
import numpy as np
import seaborn as sns
dataframe=pd.read_csv('SMSSpamCollection.txt',sep="\t",names=['label','message'])
print(dataframe.head())
from nltk import PorterStemmer
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
porter=PorterStemmer()
corpus=[]
for i in range(0,len(dataframe)):
    review=re.sub('[^a-zA-z]',' ',dataframe['message'][i])
    review=review.lower()
    review=review.split()
    review=[porter.stem(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
print(corpus)
#create bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(binary=True,max_features=100)
X=cv.fit_transform(corpus).toarray()
np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))
print(X)
print(cv.vocabulary_)
#create bag of words with Ngrams
cv1=CountVectorizer(binary=True,ngram_range=(2,3),max_features=100)
X=cv1.fit_transform(corpus).toarray()
print(X)
print(cv1.vocabulary_)