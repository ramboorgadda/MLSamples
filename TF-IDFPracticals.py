import numpy as np
import pandas as pd
import re
import nltk 
nltk.download('stopwords')
messages=pd.read_csv('SMSSpamCollection.txt',sep='\t',names=['label','message'])
print(messages.head())
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
lemma=WordNetLemmatizer()
corpus=[]
for i in range(0,len(messages)):
    review=re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review=review.lower()
    review=review.split()
    #print("split review is ",review)
    review=[lemma.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review=' '.join(review)
    corpus.append(review)
    
print(corpus)
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(max_features=100,ngram_range=(2,2))
X=tfidf.fit_transform(corpus).toarray()
np.set_printoptions(edgeitems=30, linewidth=100000, 
    formatter=dict(float=lambda x: "%.3g" % x))
print(tfidf.vocabulary_)