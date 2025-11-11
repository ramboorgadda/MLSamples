import numpy as np
import pandas as pd
df=pd.read_csv("SMSSpamCollection.txt",sep="\t",names=["label","message"])
print(df.head())
import gensim
from gensim.models import word2vec,KeyedVectors
import gensim.downloader as api
wv=api.load('word2vec-google-news-300')
print(wv['king'])
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
import re
import nltk
nltk.download('stopwords')
corpus=[]
for i in range(0,len(df)):
    review = re.sub('[^a-zA-Z]'," ",df['message'][i])
    review=review.lower()
    review=review.split()
    review=[lemmatizer.lemmatize(word) for word in review]
    review=" ".join(review)
    corpus.append(review)
print(corpus)
[[i,j,k] for i,j,k in zip(list(map(len,corpus)),corpus, df['message']) if i<1]
from nltk import sent_tokenize
from gensim.utils import simple_preprocess
words=[]
for sent in corpus:
    sent_token=sent_tokenize(sent)
    for sent in sent_token:
        words.append(simple_preprocess(sent))
print(words)
## Lets train Word2vec from scratch
import gensim
from gensim.models import word2vec
model=word2vec.Word2Vec(words)
print(model.wv.index_to_key)
def avg_word2vec(doc):
    # remove out-of-vocabulary words
    #sent = [word for word in doc if word in model.wv.index_to_key]
    #print(sent)
    
    return np.mean([model.wv[word] for word in doc if word in model.wv.index_to_key],axis=0)
                #or [np.zeros(len(model.wv.index_to_key))], axis=0)
from tqdm import tqdm
X=[]
for i in tqdm(range(len(words))):
    X.append(avg_word2vec(words[i]))
print(len(X))
X_new=np.array(X)
print(df.shape)
print(X_new.shape)
print(X_new[0].shape)
y= df[list(map(lambda x: len(x)>0,corpus))]
y=pd.get_dummies(y['label'])
y=y.iloc[:,0].values
df=pd.DataFrame()
for i in range(0,len(X)):
    df=df.append(pd.DataFrame(X[i].reshape(1,-1)),ignore_index=True)
df['Output']=y
df.dropna(inplace=True)
df.isnull().sum()
y=df['Output']
## Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20)
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))