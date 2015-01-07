# reference : http://www.nyu.edu/projects/politicsdatalab/workshops/NLTK_presentation%20_code.py
#http://arrow.dit.ie/cgi/viewcontent.cgi?article=1062&context=scschcomcon

#homework - https://78462f86-a-fa7ed8f8-s-sites.googlegroups.com/a/mohamedaly.info/www/teaching/cmp-462-spring-2013/CMP462%20HW04%20Sentiment%202.pdf?attachauth=ANoY7cpstDSqAnBAHmSSWkCQyHGH-ixfkI2OmMPAokebwzw9qVoH1i_XCMAfYVg75s_GbEeaM6uLX9Nf44Jpc8sofTtFMGEpaBn0sKpOWjFh5XaDvHUnIih06r5o3UuLrMxS2ZwFDJL4CKCgzZ1gVxE9oWApglczEXslK9Z8VYGdcqweql-ojoLMQVJDU0568rZ3HsPzKU3hWD2CuS6f3MLnG12cKdAivWee1U-i75fSrOVkDvTwhjtdKNZ1vXklwgV3OQIil7OGcMhQJgXojn97CdxF2dtSkQ%3D%3D&attredirects=0


import nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier, MaxentClassifier
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from random import shuffle
import re
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn import svm
import string

class MovieReview:
    def __init__(self):
        self.documents = [(list(movie_reviews.words(fileid)), category)
                          for category in movie_reviews.categories()
                          for fileid in movie_reviews.fileids(category)]

        shuffle(self.documents)

        self.stopset = set(stopwords.words('english'))
        self.K_FOLDS = 10
        self.fullfeatures = []
        self.features_X = []
        self.features_Y = []

        self.negative = ["wasn\'t",'don\'t','not','bad','worst','ugly','hate']
        self.end = ['\,','\.']
        self.negationFeatures = []

    def extract_features(self, document, polarity):
        #print (document)
        document_words = document
        # and not re.match(r'.*[\$\^\*\@\!\_\-\(\)\:\;\'\"\{\}\[\]].*', word.lower())
        # features = dict([(word.lower(), True) for word in document_words if
        #                  word.lower() not in self.stopset or not re.match(r'.*[\$\^\*\@\!\_\-\(\)\:\;\'\"\{\}\[\]\,].*',
        #                                                                   word.lower())])
        '''
        Made the following changes to extract negation
        https://www.englishclub.com/vocabulary/adjectives-personality-negative.htm
        '''
        features = []
        agree = True
        joined = []
        for word in document:
            if (polarity == 'pos' and word in self.negative) or agree == False:
               # print joined
                if word in string.punctuation:
                    #print('yahaaayaaa')
                    agree = True
                    line = ' '.join(joined)
                    #print(joined)
                    self.negationFeatures.extend(joined)
                    #print(self.negationFeatures)
                    del joined[:]
                    #line = ''
                else:
                    agree = False
                    joined.append(word)
                    continue
            elif agree:
                features.append((word.lower(), True))


        self.negationFeatures = list(set(self.negationFeatures))
        print self.negationFeatures
        return dict(features)

    def exploreContents(self):
        words = movie_reviews.words('pos/cv000_29590.txt')
        # print(words)
        print(self.extract_features(words,'pos'))

    def getAllFeatures(self):
        for d, c in self.documents:
            extract = self.extract_features(d,c)
            self.fullfeatures.append((extract, c))

            line = ' '.join(list(extract))
            line = self.cleanLine(line)

            self.features_X.append(line)
            self.features_Y.append(c)


            #words = ' ,'.join(self.negationFeatures)
            #words = list(set(words))
            #print(words)

            for value in self.negationFeatures:
                value = self.cleanLine(value)
                self.features_X.append(value)
                self.features_Y.append('neg')

    def cleanLine(self,text):
        line = re.sub("U+[a-zA-Z0-9]{0,10}", "", text)
        line = re.sub("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", line)
            # replace all punctuation
        line = re.sub('[^\w\s]', "", line)
        return line

    def kfoldTest(self):
        cutoff = len(self.fullfeatures) * 3 / 4
        train = self.fullfeatures[:cutoff]
        test = self.fullfeatures[cutoff:]
        X_folds = np.array_split(train, self.K_FOLDS)
        scores = list()
        for k in range(self.K_FOLDS):
            X_train = list(X_folds)
            X_test = X_train.pop(k)
            X_train = np.concatenate(X_train)
            # classifier_sub = MaxentClassifier.train(X_train, 'GIS', trace=3,
            # encoding=None, labels=None, sparse=True, gaussian_prior_sigma=0,
            # max_iter=10)
            classifier_sub = NaiveBayesClassifier.train(X_train)
            scores.append(nltk.classify.util.accuracy(classifier_sub, X_test))
            print classifier_sub.show_most_informative_features(5)
            print("=======================================")

        print("K-Fold scores done ", scores)
        print("Average ", np.mean(scores))


    def nltkClassifier(self):

        self.kfoldTest()
        cutoff = len(self.fullfeatures) * 3 / 4
        train = self.fullfeatures[:cutoff]
        test = self.fullfeatures[cutoff:]
        classifier = NaiveBayesClassifier.train(train)
        print("****************************************")
        print classifier.show_most_informative_features(10)
        print "Proper Accuracy ", nltk.classify.accuracy(classifier, test)


    def scikitKFold(self, classifier):

        X_folds = np.array_split(self.features_X, 3)
        y_folds = np.array_split(self.features_Y, 3)

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

        print("K-Fold scores @Scikit done ", scores)
        print("Average @Scikit ", np.mean(scores))

    def scikitClassifier(self):

        classifier = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('classifier', svm.SVC(kernel='linear',cache_size=4000)
            )])

        #OneVsRestClassifier(LinearSVC()) - 86.7
        #MultinomialNB(alpha=1.0,class_prior=None,fit_prior=True) - 83
        #BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True) - 79
        #svm.SVC(kernel='linear') - 87

        self.scikitKFold(classifier)

        size = len(self.features_X) * 2 / 3
        X_train = np.array(self.features_X[0:size])
        y_train = np.array(self.features_Y[0:size])

        X_test = np.array(self.features_X[size + 1: len(self.features_X)])
        y_test = np.array(self.features_Y[size + 1: len(self.features_Y)])

        classifier = classifier.fit(X_train, y_train)
        print "Scores @Scikit", classifier.score(X_test, y_test)


review = MovieReview()
#review.exploreContents()
review.getAllFeatures()
#review.nltkClassifier()
review.scikitClassifier()