words=["eating","eats","ate","eaten","eat","writing","writes","programming","programs","history"]
#porter_stemmer = nltk.PorterStemmer()
from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()
for word in words:
    print(f"Original: {word} | Porter Stemmed: {porter_stemmer.stem(word)}")
# Regular expression stemmer
from nltk.stem import RegexpStemmer
reg_stemmer = RegexpStemmer('ing$|s$|e$|able$', min=4)
print(reg_stemmer.stem("eating"))
from nltk.stem import SnowballStemmer
snowball_stemmer = SnowballStemmer("english")
words = ["eating", "eats", "ate", "eaten", "eat", "writing", "writes", "programming", "programs", "history"]
for word in words:
    print(f"Original: {word} | SnowBall Stemmed: {snowball_stemmer.stem(word)}")
    
print(f" porter stemmer: {porter_stemmer.stem('fairly')}")
print(f" porter stemmer: {porter_stemmer.stem('sportingly')}")
print(f" snowball stemmer: {snowball_stemmer.stem('fairly')}")
print(f" snowball stemmer: {snowball_stemmer.stem('sportingly')}")