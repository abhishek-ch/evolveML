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
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier

from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from nltk.stem.porter import PorterStemmer

# todo https://github.com/yogesh-kamble/kaggle-submission/blob/master/movie_review.py
class DataReader(object):
    def __init__(self):
        self.signature = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Miss.']
        # self.stop_words = list(set(stopwords.words('english')))
        self.remove_words = ["'ve", "'nt", "'ll", "n't", '...', "'re", "'d", "'n", "'m", "'em", "'til","'s"]
        self.previouswords = ['was', 'do', 'have', 'were', 'had', 'need', 'has', 'did']
        self.pattern1 = re.compile("^[.-]+\w+[.-]+$")
        self.wordPattern = re.compile("^[\w\d]*[\-\'][\w\d]+$")
        self.useless = re.compile("^[0123456789+-.,!@#$%^&*();\/|<>\"']+$")
        self.worddashword = re.compile("\w+-\w+")
        self.worddashword1 = re.compile("\w+-\w+-\w+")
        self.negative = ["wasn\'t", 'don\'t', 'not', 'bad', 'worst', 'ugly', 'hate']

        self.sents = {'0': 'neg', '1': 'someneg', '2': 'neutral', '3': 'sompos', '4': 'pos'}

        self.matchingparam = [',', '.']
        self.allPhrases = []

        self.punctuations = """!"#$%&'()*+/:;<=>?@[\]^_`{|}~"""

        self.stop_words = ['a', 'can', 'and', 'zone',  'yu', "`", "=", "your", "about", "you", "/", "\\",
                           "\*", "yet",
                           "young", "till", "written", "above", "year", "accepts", "abstract", "would", "writer",
                           "action", "world",
                           "according", "words", "with", "years", "word", "will", "without", "actually", "work",
                           "who", "an", "well", "all", "as", "be",
                           "of","to","it","in","is","that","for","movi","thi","film",
                           "hi","ha",
                           "?!?","all-tim","arni","bug-ey","war-torn","x.","tsai",
                           "bros.","xxx"]

        self.allmainwords = []
        self.minimumfreqwords = re.compile("'\d{2}s")
        self.minimumfreqwords1 = re.compile("-\w{3}-")
        self.minimumfreqwords2 = re.compile("[\\*]+")
        self._digits = re.compile('\d')
        self.stemmer = PorterStemmer()




    def countWordFreq(self, count_vectorizer, frequencies):
        word_freq_df = pd.DataFrame({'term': count_vectorizer.get_feature_names(),
                                     'occurrences': np.asarray(frequencies.sum(axis=0)).ravel().tolist()})
        word_freq_df['frequency'] = word_freq_df['occurrences'] / np.sum(word_freq_df['occurrences'])

        print word_freq_df.sort('occurrences', ascending=False).head(100)
        #wordlist = word_freq_df[word_freq_df.occurrences == 38]
        #print('length ok', wordlist)
        #return wordlist['term']

    #first word
    #capital letter always is not right 33896
    #it has significance
    #is 's is there , then previous word can be popped 151528




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

        wordlist = phrase.split()
        '''
        if len(wordlist) >0 and wordlist[len(wordlist)-1] == ".":
            _cached_.append(".")
        '''
        for word in wordlist:
            word = self.stemmer.stem(word)

            if word in self.signature:
                # print('Sig ',word)
                continue

            if word.isupper():
                # print('Upper ',word)
                continue
            if self.worddashword.match(word) or self.worddashword1.match(word):
                continue
            #if word in self.stop_words:
             #   continue
            if word in self.remove_words:
                if len(_cached_) > 0:
                    _cached_.pop()
                continue
            if self.minimumfreqwords.match(word):
                continue
            if self.useless.match(word):
                continue
            if word == "'s":
                if len(_cached_) > 0:
                    _cached_.pop()
                continue
            if bool(self._digits.search(word)):
                continue
            #if word in string.punctuation:
                #print(phrase)
            #    continue
            if word[0].isupper():
                continue

            _cached_.append(word.lower())
            # For First Worst Check

        return _cached_


    def KFOLDTEST(self, text, sent):
        k_fold = KFold(n=len(text), n_folds=6)

        pipeline = Pipeline([
            ('vectorizer', CountVectorizer(ngram_range=(1, 2), tokenizer=self.tokenize_data)),
            ('tfidf', TfidfTransformer(norm="l2", smooth_idf=False, use_idf=False)),
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




    def normalexecution(self, test_data, count_vectorizer, tfidf, classfier):
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
                error.append((phraseid_test[counter], [prediction, sent_test[counter], test_X[counter]]))

            counter += 1

        print(' Accuracy1 ', count, 'test1 ', len(test_data), ' Percentage ', float(count) / float(len(test_data)))

        for item in error:
            print(item)

    def analysis(self, testanalysis=True):
        if testanalysis:
            trainingdata, testdata = self.getTrainTestData()
        else:
            trainingdata, testdata = self.getRealData()

        aDict = {}
        for value in trainingdata:
            phrase = value.Phrase

            phrase = phrase.strip()

            aDict[phrase] = value.Sentiment

        _all_values = aDict.keys()
        _all_sentiments = aDict.values()

        self.KFOLDTEST(np.asarray(_all_values), np.asarray(_all_sentiments))

        count_vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=self.tokenize_data)
        count = count_vectorizer.fit_transform(_all_values)

        #self.countWordFreq(count_vectorizer, count)

        tfidf = TfidfTransformer(norm="l2", smooth_idf=False, use_idf=False)
        data = tfidf.fit_transform(count)

        classfier = OneVsOneClassifier(LinearSVC())
        classfier.fit(data, np.asarray(_all_sentiments))





        # Data to write the content into the CSV , for getting this comment the above to take entire training set
        #as the real data
        #along with that call the method @getRealData
        if testanalysis:
            self.normalexecution(testdata, count_vectorizer, tfidf, classfier)
        else:
            self.writeToFile(testdata, count_vectorizer, tfidf, classfier)



    def writeToFile(self, test_data, count_vectorizer, tfidf, classfier):
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
        data.append(('PhraseId', 'Sentiment'))
        i = 0
        for output in predicted:
            data.append((phraseid_test[i], output))
            i += 1

        for item in data:
            mywriter.writerow(item)

        tutorial_out.close()

    def writeToFile(self, test_data, count_vectorizer, tfidf, classfier):
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
        data.append(('PhraseId', 'Sentiment'))
        i = 0
        for output in predicted:
            data.append((phraseid_test[i], output))
            i += 1

        for item in data:
            mywriter.writerow(item)

        tutorial_out.close()


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
