# Source - training.1600000.processed.noemoticon.csv
__author__ = 'achoudhary'

##Mix and Match nltk and skicit learn
#http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html
###
# from pyspark import SparkContext

import pandas as pd
import os
import nltk
import numpy as np
import csv
import re
import collections
import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist
from random import shuffle


# fulldataframe = pd.DataFrame(columns=('emo', 'tweets'))
tweets = []
all_words = []
word_features = []
BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
emotions = ['Happy', 'Sad']

keys = []
values = []
K_FOLDS = 5;
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


            #K-Fold classification test
            #average the result of number of tests
            shuffle(trainfeats)
            X_folds = np.array_split(trainfeats, K_FOLDS)

            scores = list()
            for k in range(K_FOLDS):
                X_train = list(X_folds)
                X_test = X_train.pop(k)
                X_train = np.concatenate(X_train)
                classifier = NaiveBayesClassifier.train(X_train)
                scores.append(nltk.classify.util.accuracy(classifier,X_test))



            for i, (feats, label) in enumerate(testfeats):
                refsets[label].add(i)
                observed = classifier.classify(feats)
                testsets[observed].add(i)

            print 'Average accuracy K-Fold ',sum(scores)/float(len(scores))
            print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
            print 'Happy precision:', nltk.metrics.precision(refsets['Happy'], testsets['Happy'])
            print 'Happy recall:', nltk.metrics.recall(refsets['Happy'], testsets['Happy'])
            print 'Sad precision:', nltk.metrics.precision(refsets['Sad'], testsets['Sad'])
            print 'Sad recall:', nltk.metrics.recall(refsets['Sad'], testsets['Sad'])
            print 'Output:',nltk.classify.util.accuracy(classifier, ['He is Our To be Hanged'])
            # print 'Trust precision:', nltk.metrics.precision(refsets['Trust'], testsets['Trust'])
            # print 'Trust recall:', nltk.metrics.recall(refsets['Trust'], testsets['Trust'])
            # print 'Sad precision:', nltk.metrics.precision(refsets['Angry'], testsets['Angry'])
            # print 'Sad recall:', nltk.metrics.recall(refsets['Angry'], testsets['Angry'])
            classifier.show_most_informative_features()

        except AttributeError,err:
            print Exception, err


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

