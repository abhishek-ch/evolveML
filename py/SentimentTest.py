# Source - training.1600000.processed.noemoticon.csv
__author__ = 'achoudhary'
####
#natural Language Processing
#http://www.nltk.org/book_1ed/
###

# http://cs.brown.edu/courses/csci1951-a/assignments/assignment3/
# http://www.slideshare.net/benhealey/ben-healey-kiwipycon2011pressotexttospeech
# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/

##Follow up Code
#http://stackoverflow.com/questions/21107075/classification-using-movie-review-corpus-in-nltk-python
#
#
#Data Processing or Natural Language Processing
#http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html

##Mix and Match nltk and skicit learn
#http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
###

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

# fulldataframe = pd.DataFrame(columns=('emo', 'tweets'))
tweets = []
all_words = []
word_features = []
BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
emotions = ['Happy', 'Sad', 'Angry', 'Trust']

keys = []
values = []


class DataExtractor(object):
    big_regex = []

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


    def extractDataSetForNewModel(self):
        dataset = []
        datafiles = [{'emo': "Happy", 'name': "/positive.csv"}, {'emo': 'Sad', 'name': "/negative.csv"},
                     {'emo': 'Angry', 'name': "/anger.csv"}, {'emo': 'Trust', 'name': "/trust.csv"}]

        for value in datafiles:
            emo = value['emo']
            name = value['name']
            # print 'Emo %s File %s ' % (emo, name)
            read = self.readFile(name)
            read['emo'] = emo
            dataset.append(read)

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

    def extractLargeExcelValues(self):
        with open('D:/Work/Python/training.1600000.processed.noemoticon.csv') as csvfile:
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
dataExtractor = DataExtractor()
dataExtractor.extractDataSetForNewModel()
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
TEST DATA ACCURACY
http://scikit-learn.org/stable/tutorial/statistical_inference/model_selection.html
'''

# X_folds = np.array_split(fulldataframe['tweets'].tolist(), 3)
# y_folds = np.array_split(fulldataframe['emo'].tolist(), 3)

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
Why this Logic is priniting a huge set of perm and combination
svc = svm.SVC(C=1, kernel='linear')
kfold = cross_validation.KFold(n=len(keys), n_folds=3)
# print cross_validation.cross_val_score(svc, X_train, y_train, cv=kfold, n_jobs=-1)
'''

'''
Predicting Result or values based on machine learning
'''
# print X_test
predicted = classifier.predict(X_test1)
for item, labels in zip(X_test1, predicted):
    print '%s => %s' % (item, labels)


