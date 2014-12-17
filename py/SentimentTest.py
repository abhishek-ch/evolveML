__author__ = 'achoudhary'
# http://cs.brown.edu/courses/csci1951-a/assignments/assignment3/
#http://www.slideshare.net/benhealey/ben-healey-kiwipycon2011pressotexttospeech
# http://stevenloria.com/how-to-build-a-text-classification-system-with-python-and-textblob/
import pandas as pd
import os
import nltk
import random
from sklearn import naive_bayes
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier

tweets = []
all_words = []
word_features = []
keys = []
values = []

class DataExtractor(object):
    def readFile(self,filepath,header):
        # bigramData = sc.textFile(contentFile).cache()
        return pd.read_csv(filepath, names=[header],
                           header=0)


    def cleanTweets(self):
        pass


    def extractDataSetForNewModel(self):
        stopwords = self.readFile(os.getcwd()+"/stop-words.txt","stop")
        positiveDataSet = self.readFile(os.getcwd()+"/positive.csv","tweets")
        positiveDataSet['emo']='Positive'
        print "positiveDataSet ",len(positiveDataSet)
        negativeDataSet = self.readFile(os.getcwd()+"/negative.csv","tweets")
        negativeDataSet['emo']='Negative'
        print "negativeDataSet ",len(negativeDataSet)
        dataset = [positiveDataSet,negativeDataSet]

        for val in dataset:
            for index,row in val.iterrows():
                statement = row['tweets'].strip()
                statement = unicode(statement, errors='replace')
                keys.append(statement.lower())
                values.append(row['emo'])
                tweets.append((statement.lower(),row['emo']))
                # document = [statement.lower()
                #             for statement in row['tweets']]
                # tweets.append((document,row['emo']))

        return tweets

    def extractFeatures(self):
        stopwords = self.readFile(os.getcwd()+"/stop-words.txt","stop")
        positiveDataSet = self.readFile(os.getcwd()+"/positive.csv","tweets")
        positiveDataSet['emo']='Positive'
        print "positiveDataSet ",len(positiveDataSet)
        negativeDataSet = self.readFile(os.getcwd()+"/negative.csv","tweets")
        negativeDataSet['emo']='Negative'
        print "negativeDataSet ",len(negativeDataSet)
        stopwordList = stopwords.stop.tolist()
        dataset = [positiveDataSet,negativeDataSet]

        for val in dataset:
            for index,row in val.iterrows():
                document = [word.lower()
                            for word in row['tweets'].split()
                            if ((len(word) >= 3) and (word.lower() not in stopwordList))]
                tweets.append((document,row['emo']))

        return tweets




class FeatureModel(object):

    def getAllWords(self):
        for (word, sentiment) in tweets:
            all_words.extend(word)

        return self.get_word_features(all_words)

    def get_word_features(self,all_words):

        # words = list(set(all_words))
        # print "All Word Length ",len(words)
        wordlist = nltk.FreqDist(all_words)
        # print 'Printing ', wordlist.most_common(50)
        word_features = wordlist.keys()
        word_features = word_features[:2000]
        return word_features

    def document_features(self,document):
        document_words = set(document)
        # print "document_words ",len(all_words)
        features = {}
        for word in all_words:
            features['contains(%s)' % word] = (word in word_features)
        return features


#http://stackoverflow.com/questions/10526579/use-scikit-learn-to-classify-into-multiple-categories
#follow this up....
#http://cs.brown.edu/courses/csci1951-a/assignments/assignment3/
#
#
dataExtractor = DataExtractor()
dataExtractor.extractDataSetForNewModel()
trainpos = keys[0:2700]
trainneg = keys[2701:4000]
random.shuffle(trainpos)
random.shuffle(trainneg)
X_train = np.array(trainpos[1:10]+trainneg[1:10])
y_train = np.array(values[1:10]+values[3000:3009])


X_test = np.array(keys[500:525]+keys[2750:2760])

# print "trainpostrainpos==<<>>> ",trainpos[1:5]
# trainneg = values[2701:4000]
# random.shuffle(trainneg)
'''
X_train = np.array(["new york is a hell of a town",
                    "new york was originally dutch",
                    "the big apple is great",
                    "new york is also called the big apple",
                    "nyc is nice",
                    "people abbreviate new york city as nyc",
                    "the capital of great britain is london",
                    "london is in the uk",
                    "london is in england",
                    "london is in great britain",
                    "it rains a lot in london",
                    "london hosts the british museum",
                    "new york is great and so is london",
                    "i like london better than new york"])
y_train = [[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[0,1],[0,1]]
'''


# print X_train
# print y_train

# print train[1]
# print tweets[1]

cl = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', OneVsRestClassifier(LinearSVC()))])
cl.fit(X_train,y_train)

# print X_test
predicted = cl.predict('I love')
print("PREDICTED",predicted)
# print "Score Test ",cl.score(trainpos[30:35],trainneg[40:45])


'''

featureModel = FeatureModel()
word_features = featureModel.getAllWords()

print "ALL LENGTH ",len(all_words)
print " word_features ",len(word_features)
print "All Tweets ",len(tweets)

feature_set = nltk.classify.util.apply_features(featureModel.document_features, tweets)
print "Classifier ",len(feature_set)

cutoff = len(feature_set)*2/3
train = feature_set[:cutoff]
test = feature_set[cutoff:]


classifier = nltk.NaiveBayesClassifier.train(train)
print "Accuracy Dude ",(nltk.classify.accuracy(classifier,test))
print "Most Informative Features ",classifier.show_most_informative_features(10)
'''