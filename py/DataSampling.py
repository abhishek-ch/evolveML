__author__ = 'achoudhary'

# http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/
# http://ravikiranj.net/drupal/201205/code/machine-learning/how-build-twitter-sentiment-analyzer

# data exploration in python shell
# http://ampcamp.berkeley.edu/big-data-mini-course/data-exploration-using-spark.html

from pyspark import SparkContext
import os
import pprint
import sys
import random

sys.path.append('/usr/local/lib/python2.7/site-packages/')
import pandas as pd
import nltk
import time
import csv


class DataSampling(object):
    def loadcsv(self, filename):
        df = pd.read_csv(filename)
        return df


    def createDataframe(self, input):
        df = pd.DataFrame(input[1:], columns=['x'])
        print df.head()
        return df

    def cleanDataFrame(self, dataframe):
        # remove retweets
        dataframe = dataframe.replace("(RT|via)((?:\\b\\W*@\\w+)+)", "", regex=True)
        dataframe = dataframe.replace("@\\w+", "", regex=True)
        dataframe = dataframe.replace("http\\w+", "", regex=True)
        dataframe = dataframe.replace("U+[a-zA-Z0-9]{0,10}", "", regex=True)
        dataframe = dataframe.replace("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", regex=True)
        # string is more than 4 characters
        # dataframe = dataframe[dataframe.x.str.len() > 4]
        # replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
        dataframe = dataframe.replace('[^\w\s]', "", regex=True)
        # convert all string to lower case
        dataframe.x = [x.lower().strip() for x in dataframe.x]

        # f = open('stop-words.txt', 'r')
        # stop_words = f.readlines()

        # if I don't believe in bigrams , I can use stop words to filter out values
        # for index, row in dataframe.iterrows():
        # if row in stop_words:
        # dataframe.loc[index]

        # need to work on stop words if any only if using unigram
        return dataframe


class Frequency(object):
    def findWordFrequency(self, all_words):
        wordlist = nltk.FreqDist(all_words)
        print 'Printing ', wordlist.most_common(50)
        return wordlist.keys()


    def extractAllWords(self, df):
        all_words = []
        for index, row in df.iterrows():
            # split all the value to each specific words
            all_words.extend((''.join(row)).split())
            # print df.x
        return all_words


feature_list = []


class FeatureWork(object):
    all_words = []

    def readUnigrams(self):
        file = "/Users/abhishekchoudhary/Work/python/evolveML/py/post_neg.txt"
        # bigramData = sc.textFile(contentFile).cache()
        return pd.read_csv(file, names=['term', 'sentimentScore', 'numPositive', 'numNegative'], sep='\t',
                           header=None)


    def getFeaturedWords(self):
        unidf = self.readUnigrams()
        # create random index
        unidf = unidf.ix[random.sample(unidf.index, 100)]
        unidf = unidf.replace("@", "", regex=True)
        unidf = unidf.replace("#", "", regex=True)
        unidf = unidf.replace("http\\w+", "", regex=True)
        feature = []
        positive = []
        negative = []
        print "length f UDFFDFDFDFD ", len(unidf)
        self.all_words = unidf.term.tolist()
        for index, row in unidf.iterrows():
            try:
                val = row['term']
                pos = row['numPositive']
                neg = row['numNegative']
                if val.startswith("http\\w+"):
                    unidf.drop(index)
                else:
                    if pos >= neg:
                        positive.append(val)
                    else:
                        negative.append(val)

                feature.append([positive, 'positive'])
                feature.append([negative, 'negative'])
                # all_words = list(set(all_words))

            except AttributeError:
                unidf.drop(index)
        return feature

    def extractSentimentDataset(self, tweets):
        unidf = self.readUnigrams()
        print unidf.head()

        unidf = unidf.replace("@", "", regex=True)
        unidf = unidf.replace("#", "", regex=True)
        unidf = unidf.replace("http\\w+", "", regex=True)
        feature_list = unidf.terms.tolist()
        feature = []
        positive = []
        negative = []
        for index, row in unidf.iterrows():
            try:
                val = row['term']
                pos = row['numPositive']
                neg = row['numNegative']
                if val.startswith("http\\w+"):
                    unidf.drop(index)
                else:
                    if pos > neg:
                        positive.append(val)
                    else:
                        negative.append(val)

                feature.append((positive, 'positive'))
                feature.append((negative, 'negative'))

            except AttributeError:
                unidf.drop(index)

        return feature


    def extract_features(self, sampledata):
        document_words = set(sampledata)
        features = {}
        for word in feature_list:
            features['contains(%s)' % word] = (word in document_words)
        return features

    def readbigrams(self):
        file = "/Users/abhishekchoudhary/Downloads/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon.txt"
        # bigramData = sc.textFile(contentFile).cache()
        return pd.read_csv(file, names=['term', 'sentimentScore', 'numPositive', 'numNegative'], sep='\t',
                           header=None)


    def perform(self, tweet_df):
        # Extract feature vector for all tweets in one shote
        training_set = nltk.classify.util.apply_features(self.extract_features, tweet_df.x.tolist())
        print training_set
        return training_set


    def document_features(self, document):
        document_words = set(document)
        # print "document_words ", len(self.all_words)
        features = {}
        for word in self.all_words:
            features['contains(%s)' % word] = (word in document_words)
        return features


sc = SparkContext("yarn-client", "Data Sampling")
# contentFile = "hdfs:///stats/foo5.csv"
# logData = sc.textFile(contentFile).cache()
# sampling = DataSampling()
# df = sampling.createDataframe(logData.take(logData.count()))
# # df = sampling.loadcsv("/Users/abhishekchoudhary/Work/python/evolveML/foo5.csv")
#
# df = sampling.cleanDataFrame(df)



# print df.head()

# required to make word cloud
# frequence = Frequency()
# all_words = frequence.extractAllWords(df)
# frequence.findWordFrequency(all_words)

feature = FeatureWork()
feature_value = feature.getFeaturedWords()

print "LENGTHH ==========::::%s ----- %s " % (len(feature_value), len(feature.all_words))
# Extract feature vector for all tweets in one shot
training_set = nltk.classify.util.apply_features(feature.document_features, feature_value)

# featuresets = [(feature.document_features(d), c) for (d, c) in feature_value]
print "training_settraining_settraining_set ", len(training_set)
print "TRAINING SETETETE ", training_set[20:]
# Train the classifier
NBClassifier = nltk.NaiveBayesClassifier.train(training_set)
print "MOST ONE ", NBClassifier.show_most_informative_features(32)
