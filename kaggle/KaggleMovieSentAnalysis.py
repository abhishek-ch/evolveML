__author__ = 'achoudhary'

import os
import csv
from MovieRecord import MovieData, TestMovieData
import random
import nltk
import collections
import string
from nltk.corpus import stopwords
import re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier,OneVsOneClassifier

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold

#todo https://github.com/yogesh-kamble/kaggle-submission/blob/master/movie_review.py
class DataReader(object):
    def __init__(self):
        self.signature = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Miss.']
        # self.stop_words = list(set(stopwords.words('english')))
        self.remove_words = ["'ve", "'nt", "'ll", "n't", '...', "'re'", "'s"]
        self.previouswords = ['was', 'do', 'have', 'were', 'had', 'need', 'has', 'did']
        self.pattern1 = re.compile("^[.-]+\w+[.-]+$")
        self.wordPattern = re.compile("^[\w\d]*[\-\'][\w\d]+$")
        self.useless = re.compile("^[0123456789+-.,!@#$%^&*();\/|<>\"']+$")
        self.worddashword = re.compile("^\w+-\w+$")
        self.negative = ["wasn\'t", 'don\'t', 'not', 'bad', 'worst', 'ugly', 'hate']

        self.sents = {'0': 'neg', '1': 'someneg', '2': 'neutral', '3': 'sompos', '4': 'pos'}

        self.matchingparam = [',', '.']
        self.allPhrases = []
        self.stop_words = ['the', 'a', 'of', 'and', 'to', 'in', 'is', 'that', 'it', 'as', 'with', 'for', 'its',
                           'an', 'of the', 'film', 'this', 'movie', 'be', 'on', 'all', 'by', 'or', 'at', 'not', 'like'
            , 'you',  'more', 'his', 'are', 'has', 'so', "``"]
        # self.most_significant_words = [],

    # http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-python-parameters
    # https://github.com/rafacarrascosa/samr/blob/develop/samr/corpus.py
    def readFilelinebyline(self, DIRECTORY, file='train.tsv', movieRecordType=MovieData):
        filepath = os.path.join(DIRECTORY, file)
        if os.path.exists(filepath):
            # read the file contents seperated by tab with ignoring the first row
            # using namedtuple here to arrange data one after
            iter = csv.reader(open(filepath, 'r'), delimiter="\t")
            next(iter)  # ignore first row

            for row in iter:
                yield movieRecordType(*row)  # * passes the value as list
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


    def getRealData(self):
        datalist = []
        BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        datalist.extend(self.readFilelinebyline(BASE_DATA_PATH))

        testdata = []
        testdata.extend(self.readFilelinebyline(BASE_DATA_PATH, 'test.tsv', TestMovieData))

        return datalist, testdata


    def tokenize_data(self, phrase):
        _cached_ = []

        for word in phrase.split():
            if word in self.signature:
                # print('Sig ',word)
                continue
            if len(word) > 0 and (word[0].isupper() and len(_cached_) > 1) or word.isupper():
                # print('Upper ',word)
                continue
            if self.worddashword.match(word.lower()):
                continue
            if word.lower() in self.stop_words:
                continue
            if word.lower() in self.remove_words:
                if len(_cached_) > 0:
                    _cached_.pop()
                continue
            if self.pattern1.match(word.lower()):
                # print('Pattern ',word)
                continue
            if self.useless.match(word.lower()):
                continue
            if word in string.punctuation:
                continue

            _cached_.append(word.lower())
            # For First Worst Check

        return _cached_


    def KFOLDTEST(self, text, sent):
        k_fold = KFold(n=len(text), n_folds=6)

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=self.tokenize_data)),
            ('tfidf', TfidfTransformer(norm="l2", smooth_idf=True, use_idf=True)),
            ('classifier', OneVsOneClassifier(LinearSVC())
            )])

        scores = []
        for train_indices, test_indices in k_fold:
            # print('Train: %s | test: %s' % (train_indices, test_indices))
            train_text = text[train_indices]
            train_y = sent[train_indices]

            test_text = text[test_indices]
            test_y = sent[test_indices]

            pipeline.fit(train_text, train_y)
            score = pipeline.score(test_text, test_y)
            scores.append(score)

        score = sum(scores) / len(scores)
        print('scores ', scores, ' Score ', score)
        return score


    def countWordFreq(self,count_vectorizer,frequencies):
        word_freq_df = pd.DataFrame({'term': count_vectorizer.get_feature_names(),
                             'occurrences': np.asarray(frequencies.sum(axis=0)).ravel().tolist()})
        word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])
        print word_freq_df.sort('occurrences', ascending=False).head(20)


    def writeToFile(self,test_data,count_vectorizer,tfidf,classfier):
        test_X = []
        phraseid_test = []
        for line in test_data:
            test_X.append(line.Phrase)
            phraseid_test.append(line.PhraseId)

        example_counts = count_vectorizer.transform(test_X)
        tfidf1 = tfidf.transform(example_counts)
        predicted = classfier.predict(tfidf1)


        tutorial_out = open('sentiment.csv', 'wb')
        mywriter = csv.writer(tutorial_out)
        data = []
        data.append(('PhraseId','Sentiment'))
        i = 0
        for output in predicted:
            data.append((phraseid_test[i],output))
            i += 1

        for item in data:
             mywriter.writerow(item)

        tutorial_out.close()


    def normalexecution(self,test_data,count_vectorizer,tfidf,classfier):
        test_X = []
        sent_test = []
        phraseid_test = []
        for line in test_data:
            test_X.append(line.Phrase)
            sent_test.append(line.Sentiment)
            phraseid_test.append(line.PhraseId)

        example_counts = count_vectorizer.transform(test_X)
        tfidf1 = tfidf.transform(example_counts)
        predicted = classfier.predict(tfidf1)

        count = 0
        counter = 0
        error = []
        for prediction in predicted:
            if prediction == sent_test[counter]:
                count += 1
            else:
                error.append((phraseid_test[counter], [prediction, sent_test[counter]]))

            counter += 1

        print(' Accuracy1 ', count, 'test1 ', len(test_data), ' Percentage ', float(count) / float(len(test_data)))
        print(error)

    def analysis(self, testanalysis = True):
        if testanalysis:
            trainingdata,testdata = self.getTrainTestData()
        else:
            trainingdata,testdata = self.getRealData()

        aDict = {}
        for value in trainingdata:
            phrase = value.Phrase

            if phrase.startswith(tuple(self.matchingparam)):
                phrase = phrase[1:]
            if phrase.endswith(tuple(self.matchingparam)):
                phrase = phrase[0:len(phrase) - 1]

            phrase = phrase.strip()

            aDict[phrase] = value.Sentiment

            '''
            if not aDict.has_key(phrase):
                aDict[phrase] = value.Sentiment
            elif not aDict[phrase] == value.Sentiment:
                print(value.PhraseId)
            '''

        _all_values = aDict.keys()
        _all_sentiments = aDict.values()

        #self.KFOLDTEST(np.asarray(_all_values), np.asarray(_all_sentiments))

        count_vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=self.tokenize_data)
        count = count_vectorizer.fit_transform(_all_values)

        self.countWordFreq(count_vectorizer,count)

        tfidf = TfidfTransformer(norm="l2", smooth_idf=True, use_idf=True)
        data = tfidf.fit_transform(count)

        classfier = OneVsRestClassifier(LinearSVC())
        classfier.fit(data, _all_sentiments)


        # Data to write the content into the CSV , for getting this comment the above to take entire training set
        #as the real data
        #along with that call the method @getRealData
        if testanalysis:
            self.normalexecution(testdata,count_vectorizer,tfidf,classfier)
        else:
            self.writeToFile(testdata,count_vectorizer,tfidf,classfier)


    def getTrainTestData(self):
        reader = DataReader()
        train_data, test_data = reader.getData()
        print('Train ', len(train_data), ' Test ', len(test_data))
        return train_data, test_data


if __name__ == '__main__':
    reader = DataReader()
    #train_data, test_data = reader.getTrainTestData()
    reader.analysis(True)
    #reader.tryanother(train_data)
