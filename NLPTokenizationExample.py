corpus="""Hello welcome,to Rama krishna's tutorials.
please do watch entire course! to become an expert in NLP."""
print(corpus)
# Tokenization
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize, sent_tokenize,wordpunct_tokenize
documents=sent_tokenize(corpus)
print(type(documents))
print(f" sentence tokenized are {documents}")
# convert paragraph to words
words=word_tokenize(corpus)
print(type(words))
print(f" words tokenized are {words}")
wordpunct_tokenizes=wordpunct_tokenize(corpus)
print(type(wordpunct_tokenizes))
print(f"wordpunct tokenized are {wordpunct_tokenizes}")
from nltk.tokenize import TreebankWordTokenizer
TreebankWordTokenizer=TreebankWordTokenizer()
treebank_words=TreebankWordTokenizer.tokenize(corpus)
print(type(treebank_words))
print(f"tree bank words are : {treebank_words}")