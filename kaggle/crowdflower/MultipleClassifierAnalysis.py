__author__ = 'abhishekchoudhary'
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
import os
import pandas as pd
import re, string
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn import decomposition, pipeline
from sklearn.decomposition import TruncatedSVD,ProjectedGradientNMF
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.svm import SVC
from sklearn.ensemble import *

#`@linked http://scikit-learn.org/stable/auto_examples/text/document_classification_20newsgroups.html#example-text-document-classification-20newsgroups-py

def tokenize(data):
    _cached_ = []
    data = BeautifulSoup(data).get_text()
    for word in data.split():
        word = word.lower()
        word = stemmer.stem(word)
        if word not in cachedStopWords and word not in string.punctuation and len(word) > 3:
            word = re.sub("http\\w+", "", word)
            word = re.sub("U+[a-zA-Z0-9]{0,10}", "", word)
            word = re.sub("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", word)
            #print(word)
            _cached_.append(word)

    return _cached_


# Benchmark classifiers
def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)

    pipelineGlobal = pipeline.Pipeline([('svd', TruncatedSVD()),
						 ('scl', StandardScaler()),
                	     ('classifier', clf)
                        ])

    t0 = time()
    pipelineGlobal.fit(X_train, datatest)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = pipelineGlobal.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    '''
    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))
    '''
    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time


if __name__ == '__main__':
    stemmer = PorterStemmer()
    cachedStopWords = stopwords.words("english")
    BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..', 'crowdflower'))
    # Use Pandas to read in the training and test data
    train = pd.read_csv(BASE_DATA_PATH + "/train.csv").fillna("")
    test = pd.read_csv(BASE_DATA_PATH + "/test.csv").fillna("")

    # we dont need ID columns
    idx = test.id.values.astype(int)
    y = train.median_relevance.values
    # drop the column
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    print("LENGTH ", len(train.median_relevance))
    # train.product_description = list(map(lambda row: BeautifulSoup(row).get_text(), train.product_description))
    out = pd.DataFrame(columns={'text','target'})
    out['text'] = train['query'].map(str)+' '+train['product_title'].map(str)+' '+train['product_description']
    out['target'] = y
    #print(out)
    #pd.concat(train['query'],train['product_title'],train['product_description'])
    traindata = list(
        train.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1))

    msk = np.random.rand(len(out)) < 0.7
    training,testing = out[msk],out[~msk]
    print len(training),len(testing)

    datatrain = training['text']
    datatest = training['target']
    y_train = testing['text']
    y_test =testing['target']

    vectorizer = TfidfVectorizer(min_df=3, max_features=None,tokenizer=tokenize,
                                 strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                                 ngram_range=(1, 3), use_idf=True, smooth_idf=1, sublinear_tf=1,
                                 stop_words='english')


   # print(datatrain[1:5])
    X_train = vectorizer.fit_transform(datatrain)
    X_test = vectorizer.transform(y_train)

    adaclassifier = AdaBoostClassifier(
        n_estimators = 40,
        learning_rate = 0.85,
        base_estimator = ExtraTreesClassifier(
            n_estimators = 400,
            min_samples_split = 100,
            verbose = 1,
            n_jobs = -1))

    results = []
    for clf, name in (
            (RidgeClassifier(alpha=1.0, class_weight=None, copy_X=True, fit_intercept=True,
            max_iter=None, normalize=False, solver='cholesky', tol=0.01), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest"),
            (adaclassifier, "Extra Trees"),
            (SVC(),'SVM')):
        print('=' * 80)
        print(name)
        results.append(benchmark(clf))


    print(results)

