__author__ = 'abhishekchoudhary'

# https://spark.apache.org/docs/1.0.2/running-on-yarn.html
# http://spark.apache.org/docs/1.0.1/configuration.html
# https://spark.apache.org/docs/1.1.0/configuration.html

# load csv file
#http://stackoverflow.com/questions/24299427/how-do-i-convert-csv-file-to-rdd

from pyspark import SparkConf, SparkContext
from Util import Ready
import pandas as pd
import pandas as pd
import os
import nltk
import numpy as np
import csv
import re
import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from random import shuffle


K_FOLDS = 5
stopset = set(stopwords.words('english'))


class PrepareData(object):
    def prepare(self):
        # training.1600000.processed.noemoticon
        # testdata.manual.2009.06.14
        filePath = "/Users/abhishekchoudhary/Work/python/training.1600000.processed.noemoticon.csv"
        util = Ready()
        df = util.readFile(filePath, header=["polarity", "userid", "date", "none", "username", "tweets"])
        # df = pd.DataFrame(df.tweets)
        df = util.cleanDataFrame(df)
        print df.head(5)
        return df


    def word_feature(self, words):

        # stopwords filtering did improvise the accuracy , so I better keep it
        # features = dict([(word.lower(), True) for word in words])
        #
        features = dict([(word.lower(), True) for word in words if word.lower() not in stopset])

        return features


    '''
    this method is used to analytics on statements
    '''
    # http://andybromberg.com/sentiment-analysis-python/
    #http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/
    def unigramAnalysis(self, word_extract_feature, df):

        trainfeats = []
        testfeats = []
        dataset = []
        features = []

        for index, row in df.iterrows():
            statement = row['tweets']
            emotion = row['polarity']
            # features.append((word_extract_feature(statement.split()), emotion))
            features.append((word_extract_feature(statement.split()), emotion))
            # dataset = [(word_extract_feature(words),emotion) for words in statement.split()]
            # if index == 10:
            #     break


        # features = [(word_extract_feature(statement.split()), emo) for statement, emo in df['tweets'], df['polarity']]
        # dataset.append(features)
        print("Appended all Dataset features")
        shuffle(features)
        # for data in dataset:
        cutoff = len(features) * 3 / 4
        trainfeats = features[:cutoff]
        testfeats = features[cutoff:]

        # print("End of Train and Test Dataset division ",trainfeats)

        try:
            print("Start Classifying or feature extraction")
            classifier = NaiveBayesClassifier.train(trainfeats)
            print("End of Main TraininG Classifiers")
            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)


            #K-Fold classification test
            #average the result of number of tests
            print("Shuffling of entire Training Set")
            shuffle(trainfeats)
            print("Shuffling Done")
            X_folds = np.array_split(trainfeats, K_FOLDS)

            scores = list()
            for k in range(K_FOLDS):
                X_train = list(X_folds)
                X_test = X_train.pop(k)
                X_train = np.concatenate(X_train)
                # classifier_sub = MaxentClassifier.train(X_train, 'GIS', trace=3,
                #                                         encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0,
                #                                         max_iter=10)
                classifier_sub = NaiveBayesClassifier.train(X_train)
                scores.append(nltk.classify.util.accuracy(classifier_sub, X_test))

            print("K-Fold scores done ", scores)

            for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)

            print 'Average accuracy K-Fold ', sum(scores) / float(len(scores))
            print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
            print 'Happy precision:', nltk.metrics.precision(refsets['Happy'], testsets['Happy'])
            print 'Happy recall:', nltk.metrics.recall(refsets['Happy'], testsets['Happy'])
            print 'Sad precision:', nltk.metrics.precision(refsets['Sad'], testsets['Sad'])
            print 'Sad recall:', nltk.metrics.recall(refsets['Sad'], testsets['Sad'])
            # print 'Output:',nltk.classify.util.accuracy(classifier, ['He is Our To be Hanged'])
            # print 'Trust precision:', nltk.metrics.precision(refsets['Trust'], testsets['Trust'])
            # print 'Trust recall:', nltk.metrics.recall(refsets['Trust'], testsets['Trust'])
            # print 'Sad precision:', nltk.metrics.precision(refsets['Angry'], testsets['Angry'])
            # print 'Sad recall:', nltk.metrics.recall(refsets['Angry'], testsets['Angry'])
            classifier.show_most_informative_features(10)

        except AttributeError, err:
            print Exception, err


conf = (SparkConf().setMaster("yarn-client").setAppName("Long DataSet In YARN").set("spark.executor.memory", "1g"))
sc = SparkContext(conf=conf)
data = PrepareData()
df = data.prepare()
data.unigramAnalysis(data.word_feature, df)
sc.stop()