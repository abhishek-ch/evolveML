# Source - training.1600000.processed.noemoticon.csv
__author__ = 'achoudhary'
# ###
# natural Language Processing
# http://www.nltk.org/book_1ed/
# ##

# http://cs.brown.edu/courses/csci1951-a/assignments/assignment3/
# http://www.slideshare.net/benhealey/ben-healey-kiwipycon2011pressotexttospeech
# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/

# #Follow up Code
#http://stackoverflow.com/questions/21107075/classification-using-movie-review-corpus-in-nltk-python
#
#
#Data Processing or Natural Language Processing
#http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

##Mix and Match nltk and skicit learn
#http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
###
# from pyspark import SparkContext

import pandas as pd
import os
import nltk
import random
from random import sample
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn import cross_validation
import numpy as np
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
import csv
import re
from collections import defaultdict

import collections, itertools
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

# from pyspark.mllib.classification import NaiveBayes
# from pyspark import SparkContext


# fulldataframe = pd.DataFrame(columns=('emo', 'tweets'))
tweets = []
all_words = []
word_features = []
BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
emotions = ['Happy', 'Sad']

keys = []
values = []

stopset = set(stopwords.words('english'))


class DataExtractor(object):
    big_regex = []
    bestwords = set()

    def __init__(self):
        stopwords = self.readFile("/stop-words.txt", "stop")
        # regex declaration to remove stop words
        self.big_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stopwords)), re.IGNORECASE)


    def readFile(self, filepath, header="tweets"):
        # bigramData = sc.textFile(contentFile).cache()
        return pd.read_csv(BASE_DATA_PATH + filepath, names=[header],
                           header=0)


    def cleanTweets(self):
        pass


    def getAllDataSet(self):
        dataset = []
        datafiles = [{'emo': 'Sad', 'name': "/negative.csv"}, {'emo': 'Trust', 'name': "/trust.csv"},
                     {'emo': 'Angry', 'name': "/anger.csv"}, {'emo': "Happy", 'name': "/positive.csv"}]

        for value in datafiles:
            emo = value['emo']
            name = value['name']
            # print 'Emo %s File %s ' % (emo, name)
            read = self.readFile(name)
            read['emo'] = emo
            dataset.append(read)

        return dataset


    def extractDataSetForNewModel(self):
        dataset = self.getAllDataSet()

        # i = 0
        for val in dataset:
            for index, row in val.iterrows():
                statement = row['tweets'].strip()
                statement = unicode(statement, errors='replace')
                statement = statement.lower()
                #removed all the stop_words from the statement
                statement = self.big_regex.sub("", statement)
                ##remove unnecessary white-space in-between string
                statement = " ".join(statement.split())
                tweets.append((statement, row['emo']))
                # fulldataframe.loc[i] = [row['emo'], statement]
                # keys.append(statement)
                # values.append(row['emo'])
                # i += 1

        return tweets


    def word_feature(self, words):

        #stopwords filtering did improvise the accuracy , so I better keep it
        # features = dict([(word.lower(), True) for word in words])
        #
        features = dict([(word.lower(), True) for word in words if word.lower() not in stopset])

        return features


    '''
    this method is used to analytics on statements
    '''
    #http://andybromberg.com/sentiment-analysis-python/
    #http://streamhacker.com/2010/06/16/text-classification-sentiment-analysis-eliminate-low-information-features/
    def unigramAnalysis(self, word_extract_feature):

        #Dataset on Anger and Trust is extremely poor
        #it ruined my existing dataset as well , so I will avoid them as of now
        datafiles = [{'emo': "Sad", 'name': "/negative.csv"}, {'emo': "Happy", 'name': "/positive.csv"}
                      # ,{'emo': 'Happy', 'name': "/trust.csv"}, {'emo': 'Sad', 'name': "/anger.csv"}
        ]

        trainfeats = []
        testfeats = []
        dataset = []

        for value in datafiles:
            emo = value['emo']
            name = value['name']
            read = self.readFile(name)
            read['emo'] = emo
            features = [(word_extract_feature(statement.split()), emo) for statement in read['tweets']]



            dataset.append(features)

        for data in dataset:
            cutoff = len(data) * 3 / 4
            trainfeats = trainfeats + data[:cutoff]
            testfeats = testfeats + data[cutoff:]

        try:
            classifier = NaiveBayesClassifier.train(trainfeats)
            refsets = collections.defaultdict(set)
            testsets = collections.defaultdict(set)

            for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)

            print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
            print 'Happy precision:', nltk.metrics.precision(refsets['Happy'], testsets['Happy'])
            print 'Happy recall:', nltk.metrics.recall(refsets['Happy'], testsets['Happy'])
            print 'Sad precision:', nltk.metrics.precision(refsets['Sad'], testsets['Sad'])
            print 'Sad recall:', nltk.metrics.recall(refsets['Sad'], testsets['Sad'])
            # print 'Trust precision:', nltk.metrics.precision(refsets['Trust'], testsets['Trust'])
            # print 'Trust recall:', nltk.metrics.recall(refsets['Trust'], testsets['Trust'])
            # print 'Sad precision:', nltk.metrics.precision(refsets['Angry'], testsets['Angry'])
            # print 'Sad recall:', nltk.metrics.recall(refsets['Angry'], testsets['Angry'])
            classifier.show_most_informative_features()

        except AttributeError:
            pass


    def best_word_feats(self, words):
        return dict([(word, True) for word in words if word in self.bestwords])

    def best_bigram_word_feats(self, words, score_fn=BigramAssocMeasures.chi_sq, n=500):
        # print "OKOOK ",words
        #unique words within a list will be entertained else its trouble
        words = list(set(words))
        if len(words) > 0:
            # bigram_measures = nltk.collocations.BigramAssocMeasures()
            bigram_finder = BigramCollocationFinder.from_words(words)
            # print bigram_finder.nbest(bigram_measures.pmi, 10)
            try:
                bigrams = bigram_finder.nbest(score_fn, n)
                d = dict([(bigram, True) for bigram in bigrams])
                d.update(self.best_word_feats(words))
                return d
            except ZeroDivisionError:
                print "AB_CD_EF"
        else:
            print("Found Irrelevant list with One Word")


    def bigramAnalysis(self):


        label_word_fd = ConditionalFreqDist()
        word_fd = FreqDist()

        datafiles = [{'emo': "Sad", 'name': "/negative.csv"}, {'emo': "Happy", 'name': "/positive.csv"}
                     # , {'emo': 'Happy', 'name': "/trust.csv"}, {'emo': 'Sad', 'name': "/anger.csv"}
        ]

        for value in datafiles:
            emo = value['emo']
            name = value['name']
            read = self.readFile(name)
            normalized_sentences = [s.lower() for s in read['tweets']]

            for statement in normalized_sentences:
                for word in statement.split():
                    wor = word.lower()
                    if word not in stopset:
                        word_fd[word] += 1
                        label_word_fd[emo][word] += 1
                        # word_fd.inc(word.lower())

        word_scores = {}
        pos_word_count = label_word_fd['Happy'].N()
        neg_word_count = label_word_fd['Sad'].N()

        total_word_count = word_fd.N()

        for word, freq in word_fd.iteritems():
            pos_score = BigramAssocMeasures.chi_sq(label_word_fd['Happy'][word],
                                                   (freq, pos_word_count), total_word_count)
            neg_score = BigramAssocMeasures.chi_sq(label_word_fd['Sad'][word],
                                                   (freq, neg_word_count), total_word_count)
            word_scores[word] = pos_score + neg_score

        best = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:500]
        self.bestwords = set([w for w, s in best])

        print("\n\nevaluating best word features")
        self.unigramAnalysis(self.best_word_feats)

        print("\n\nBigram + bigram chi_sq word ")
        self.unigramAnalysis(self.best_bigram_word_feats)

        # print "pos_score %s  neg_score %s best %s" % (pos_score, neg_score, self.bestwords)
        # print "Emotion counts Happy %s  Sad %s Total Wordlist %s" % (pos_word_count, neg_word_count, total_word_count)
        # print "word Frequency ", word_fd.most_common(10)


    def extractLargeExcelValues(self):
        with open(
                '/Users/abhishekchoudhary/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                statement = unicode(row[5], errors='replace')
                statement = statement.lower()
                # statement = statement.replace("!@#$%^&*()[]{};:,./<>?\|`~-=_+", " ")
                statement = re.sub(r"@/?\w+", "", statement)
                statement = re.sub('[^A-Za-z0-9]+', ' ', statement)
                statement = self.big_regex.sub("", statement)
                ##remove unnecessary white-space in-between string
                statement = " ".join(statement.split())
                tweets.append((statement, 'Happy' if row[0] == '0' else 'Sad'))
                # print "row %s " %(statement)
        return tweets

    def extractFeatures(self):
        stopwords = self.readFile(os.getcwd() + "/stop-words.txt", "stop")
        positiveDataSet = self.readFile(os.getcwd() + "/positive.csv", "tweets")
        positiveDataSet['emo'] = 'Positive'
        print "positiveDataSet ", len(positiveDataSet)
        negativeDataSet = self.readFile(os.getcwd() + "/negative.csv", "tweets")
        negativeDataSet['emo'] = 'Negative'
        print "negativeDataSet ", len(negativeDataSet)
        stopwordList = stopwords.stop.tolist()
        dataset = [positiveDataSet, negativeDataSet]

        for val in dataset:
            for index, row in val.iterrows():
                document = [word.lower()
                            for word in row['tweets'].split()
                            if ((len(word) >= 3) and (word.lower() not in stopwordList))]
                tweets.append((document, row['emo']))

        return tweets


class FeatureModel(object):
    def getAllWords(self):
        for (word, sentiment) in tweets:
            all_words.extend(word)

        return self.get_word_features(all_words)

    def get_word_features(self, all_words):

        # words = list(set(all_words))
        # print "All Word Length ",len(words)
        wordlist = nltk.FreqDist(all_words)
        # print 'Printing ', wordlist.most_common(50)
        word_features = wordlist.keys()
        word_features = word_features[:2000]
        return word_features

    def document_features(self, document):
        document_words = set(document)
        # print "document_words ",len(all_words)
        features = {}
        for word in all_words:
            features['contains(%s)' % word] = (word in word_features)
        return features


# http://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
#follow this up....
#http://cs.brown.edu/courses/csci1951-a/assignments/assignment3/
#
#

# sc = SparkContext("yarn-client", "Data Sampling")

dataExtractor = DataExtractor()
print("evaluating single word features")
dataExtractor.unigramAnalysis(dataExtractor.word_feature)
dataExtractor.bigramAnalysis()


# dataExtractor.extractDataSetForNewModel()



# dataExtractor.extractLargeExcelValues()

X_test1 = np.array([
    'A justice system that protects serial murderers, rapists & gangsters & punishes innocents is condemned to spiral into chaos',
    '6 devastating images of the aftermath',
    'hello welcome to new york. enjoy it here and london too',
    'I will never forget the black boots',
    'He is Our To be Hanged',
    'it rains a lot in london',
    'I am happy to be with you',
    'what is going on here',
    'Ohh My God , that was a bad day',
    'I am going to kill you',
    'probably what youre not doing right',
    'I am not good to be back',
    "My dad says this product is terrible, but I disagree",
    'my life is miserable',
    'Unfortunately the code isn\'t open source'])

'''
np.random.shuffle(tweets)

for key, value in tweets:
    keys.append(key)
    values.append(value)

size = len(keys) * 1 / 2

X_train = np.array(keys[0:size])
y_train = np.array(values[0:size])

X_test = np.array(keys[size + 1: len(keys)])
y_test = np.array(values[size + 1: len(keys)])

print  "CXONTRIBUTION trainX %s trainY %s testX %s testY %s " % (len(X_train), len(y_train), len(X_test), len(y_test))


##read more about pipeline
#http://scikit-learn.org/stable/modules/pipeline.html
#More about OneVsRestCl..
#http://scikit-learn.org/stable/modules/multiclass.html
classifier = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('classifier', OneVsRestClassifier(LinearSVC())
    )])

#For my data set Bernoulli stood the poorest but anyway all of them are not worth
#OneVsRestClassifier(LinearSVC()) - gives 93%
#MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True) - gives 79%
#BernoulliNB(binarize=None) => 44% (unexpected bad)
# classifier = MultinomialNB() => 70%



classifier = classifier.fit(X_train, y_train)
print "Scores ", classifier.score(X_test, y_test)
'''

'''
TEST DATA ACCURACY
http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
'''

# X_folds = np.array_split(fulldataframe['tweets'].tolist(), 3)
# y_folds = np.array_split(fulldataframe['emo'].tolist(), 3)

'''
X_folds = np.array_split(X_test, 3)
y_folds = np.array_split(y_test, 3)

scores = list()
for k in range(3):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train)
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    clsf = classifier.fit(X_train, y_train)

    scores.append(clsf.score(X_test, y_test))


    # predicted = clsf.predict(X_test1)
    # for item, labels in zip(X_test1, predicted):
    #     print '%s => %s =K= %s' % (item, labels,k)

print(scores)
'''
'''
Why this Logic is priniting a huge set of perm and combination
svc = svm.SVC(C=1, kernel='linear')
kfold = cross_validation.KFold(n=len(keys), n_folds=3)
# print cross_validation.cross_val_score(svc, X_train, y_train, cv=kfold, n_jobs=-1)
'''

'''
Predicting Result or values based on machine learning
'''
'''
# print X_test
predicted = classifier.predict(X_test1)
for item, labels in zip(X_test1, predicted):
    print '%s => %s' % (item, labels)

'''
