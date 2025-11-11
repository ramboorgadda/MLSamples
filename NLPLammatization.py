import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
words=["eating","eats","ate","eaten","eat","writing","writes","programming","programs","history"]
for word in words:
    print(f"Original: {word} | Lemmatized: {lemmatizer.lemmatize(word,pos='v')}")
print(f"Lemmatized 'better': {lemmatizer.lemmatize('better', pos='a')}")
print(f"Lemmatized 'running': {lemmatizer.lemmatize('running', pos='v')}")
print(f"Lemmatized 'ran': {lemmatizer.lemmatize('goes', pos='v')}")