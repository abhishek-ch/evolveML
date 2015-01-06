# reference : http://www.nyu.edu/projects/politicsdatalab/workshops/NLTK_presentation%20_code.py
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from random import shuffle
import re

class MovieReview:
    def __init__(self):
        self.documents = [(list(movie_reviews.words(fileid)), category)
                          for category in movie_reviews.categories()
                          for fileid in movie_reviews.fileids(category)]

        shuffle(self.documents)

        self.stopset = set(stopwords.words('english'))


    def extract_features(self, document):
        document_words = set(document)
        #and not re.match(r'.*[\$\^\*\@\!\_\-\(\)\:\;\'\"\{\}\[\]].*', word.lower())
        features = dict([(word.lower(), True) for word in document_words if
                         word.lower() not in self.stopset and not re.match(r'.*[\$\^\*\@\!\_\-\(\)\:\;\'\"\{\}\[\]].*', word.lower())])

        return features

    def exploreContents(self):
        words = movie_reviews.words('pos/cv957_8737.txt')
       # print(words)
        print(self.extract_features(words))

review = MovieReview()
review.exploreContents()


