__author__ = 'achoudhary'

import os
import csv
from MovieRecord import MovieSingleData, MovieData
import random
import nltk
import collections
import string
from nltk.corpus import stopwords
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.svm import SVC
import numpy as np


class DataReader(object):
    def __init__(self):
        self.signature = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Miss.']
        self.stop_words = list(set(stopwords.words('english')))
        self.remove_words = ["'ve", "'nt", "'ll", "n't", '...', "'re'", "'s"]
        self.previouswords = ['was', 'do', 'have', 'were', 'had', 'need', 'has', 'did']
        self.pattern1 = re.compile("^[.-]+\w+[.-]+$")
        self.wordPattern = re.compile("^[\w\d]*[\-\'][\w\d]+$")
        self.useless = re.compile("^[0123456789+-.,!@#$%^&*();\/|<>\"']+$")
        self.worddashword = re.compile("^\w+-\w+$")
        self.negative = ["wasn\'t", 'don\'t', 'not', 'bad', 'worst', 'ugly', 'hate']

        self.sents = {'0': 'neg', '1': 'someneg', '2': 'neutral', '3': 'sompos', '4': 'pos'}
        # self.most_significant_words = []

    # http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-python-parameters
    # https://github.com/rafacarrascosa/samr/blob/develop/samr/corpus.py
    def readFilelinebyline(self, DIRECTORY):
        filepath = os.path.join(DIRECTORY, 'train.tsv')
        if os.path.exists(filepath):
            # read the file contents seperated by tab with ignoring the first row
            # using namedtuple here to arrange data one after
            iter = csv.reader(open(filepath, 'r'), delimiter="\t")
            next(iter)  # ignore first row

            for row in iter:
                yield MovieData(*row)  # * passes the value as list
                # yield MovieSingleData(row[0],row[1],row[2],row[3])

        else:
            print 'Sorry, but file path is invalid...'

    def getData(self, datalist=[]):
        # get current project root -> one step up -> get data dir -> get the file
        BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        datalist.extend(self.readFilelinebyline(BASE_DATA_PATH))

        # shuffle the entire dataset
        random.shuffle(datalist)
        # diviide the dataset in test and training data
        datalength = len(datalist)
        traincount = datalength - datalength / 6

        train_data = datalist[:traincount]
        test_data = datalist[traincount:]

        return train_data, test_data

    def verifyWord(self, word):
        if word in self.signature:
            return False
        if len(word) > 0 and (word[0].isupper()) or word.isupper():
            return False
        if self.worddashword.match(word.lower()) or word in string.punctuation:
            return False
        if word.lower() in self.stop_words:
            return False
        if word.lower() in self.remove_words:
            return False
        if self.pattern1.match(word.lower()):
            return False
        if self.useless.match(word.lower()):
            return False

        return True

    def cleanData(self, data=[]):
        specific_words_dict = collections.defaultdict(list)
        all_words = []
        skip = False
        possentiments = ['2', '3', '4']
        previous = ""

        if len(data) < 2:
            print 'Sorry but its too less data for ML'
        else:
            for line in data:
                looper = False
                phrase = line.Phrase
                sentiment = line.Sentiment
                phraseid = line.PhraseId
                i = 0
                _cached_ = []
                _negativecached_ = []


                previousword = ''
                for word in phrase.split():
                    # extract name and name honouring
                    # negation foundation
                    if word.lower() in self.remove_words and sentiment in possentiments and previous in self.previouswords:
                        #print(' word ', word, ' prev ', previous, ' sentiment ', sentiment)
                        looper = True
                        if self.verifyWord(previous):
                            _negativecached_.append(previous)
                            all_words.append(word.lower())

                    elif looper:
                        if word in string.punctuation:
                            looper = False
                        else:
                            if self.verifyWord(word):
                                #print('sentimentsentiment ',sentiment,' ph ',phraseid)
                                _negativecached_.append(word)
                                all_words.append(word.lower())
                    else:
                        if self.verifyWord(word):
                            _cached_.append(word)
                            all_words.append(word.lower())

                    previous = word
                line = ' '.join(_cached_)
                specific_words_dict[sentiment].append(line)

                if len(_negativecached_)>0:
                    #print('_negativecached_ ',_negativecached_)
                    negline = ' '.join(_negativecached_)
                    specific_words_dict[str(int(sentiment)-1)].append(negline)

        return specific_words_dict, all_words


    # find the frequency of the all word list
    def wordFrequency(self, data=[]):
        if len(data) < 2:
            return ['Sorry but its too less data for ML']
        else:
            specific_words_dict, all_words = self.cleanData(data)
            word_features = nltk.FreqDist(all_words)
            print(word_features.most_common(10))
            return specific_words_dict, all_words, word_features.keys()[:6000]


    # extracting features from the dataset extracted
    # it currently checks the availability among the most popular words in the list
    def feature_extraction(self, specific_words_dict, most_significant_words):
        features_X = []
        features_Y = []
        for sentiment, values in specific_words_dict.iteritems():
            for line in values:
                features_X.append(line)
                features_Y.append(self.sents[sentiment])
                '''
                for word in line.split():
                    if word in most_significant_words:
                        features_X.append(word)
                        features_Y.append(sentiment)
                '''
        return features_X, features_Y


    def kfoldClassification(self, X, y, classifier):
        # Let's do a 2-fold cross-validation of the SVC estimator
        print 'Cross validation ', cross_val_score(classifier, X, y, cv=10, n_jobs=1)


    def classifier(self, features_X, features_Y, test_data):
        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', OneVsRestClassifier(LinearSVC())
            )])



        # OneVsRestClassifier(LinearSVC()) - 92.3
        # MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True) - 86.7
        #generally sentiment analysis works better with BernoulliNB because of boolean nature
        #BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True) - 84.5
        #svm.SVC(kernel='linear', cache_size=5000) - 92.3

        size = len(features_X)
        X_train = np.array(features_X[0:size])
        y_train = np.array(features_Y[0:size])

        #print(' feaxtureX ',X_train[1:4],' length ',len(X_train), ' Y ',y_train[58000:58010])
        self.kfoldClassification(features_X, features_Y, classifier)
        classifier = classifier.fit(X_train, y_train)
        count = 0

        error = []
        for line in test_data:
            phrase = line.Phrase
            sentiment = line.Sentiment
            PhraseId = line.PhraseId
            predicted = classifier.predict([phrase])
            #print(' predicted ',predicted[0],' Sent ',sentiment)
            if predicted == self.sents[sentiment]:
                count += 1
            else:
                error.append(PhraseId)

        print(' Accuracy ', count, 'test ', len(test_data) , 'percetage ',float(count)/float(len(test_data)))
        print('error ', len(error))
        error.sort()
        print(error)


    def getTrainTestData(self):
        reader = DataReader()
        train_data, test_data = reader.getData()
        print('Train ', len(train_data), ' Test ', len(test_data))
        return train_data, test_data


if __name__ == '__main__':
    reader = DataReader()
    train_data, test_data = reader.getTrainTestData()
    specific_words_dict, all_words, reader.most_significant_words = reader.wordFrequency(list(train_data))
    features_X, features_Y = reader.feature_extraction(specific_words_dict, reader.most_significant_words)
    reader.classifier(features_X, features_Y, test_data)
    #print(features_X[1:5])__author__ = 'achoudhary'
