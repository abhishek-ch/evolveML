__author__ = 'abhishekchoudhary'
import sys

# sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
#http://www.nltk.org/book/ch06.html


import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
# nltk.download()

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# for list,key in documents:
#     for word in list:
#         print word
#     print key

for fileid in movie_reviews.words('pos/cv994_12270.txt'):
    print "WTH ",fileid



all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = all_words.keys()[:2000]
# print word_features

def document_features(document):
    document_words = set(document)
    print "document_words ",len(word_features)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

# print(document_features(movie_reviews.words('pos/cv957_8737.txt')))

featuresets = [(document_features(d), c) for (d,c) in documents]
train_set, test_set = featuresets[100:], featuresets[:100]
print featuresets[1:2]