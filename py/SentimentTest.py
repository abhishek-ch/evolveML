__author__ = 'achoudhary'
# http://cs.brown.edu/courses/csci1951-a/assignments/assignment3/
#http://www.slideshare.net/benhealey/ben-healey-kiwipycon2011pressotexttospeech
import pandas as pd
import os
import nltk

tweets = []
all_words = []
word_features = []

class DataExtractor(object):
    def readFile(self,filepath,header):
        # bigramData = sc.textFile(contentFile).cache()
        return pd.read_csv(filepath, names=[header],
                           header=0)


    def cleanTweets(self):
        pass


    def extractFeatures(self):
        stopwords = self.readFile(os.getcwd()+"/stop-words.txt","stop")
        positiveDataSet = self.readFile(os.getcwd()+"/positive.csv","tweets")
        positiveDataSet['emo']='Positive'

        negativeDataSet = self.readFile(os.getcwd()+"/negative.csv","tweets")
        negativeDataSet['emo']='Negative'

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
        print all_words[1:5]
        wordlist = nltk.FreqDist(all_words)
        # print 'Printing ', wordlist.most_common(50)
        word_features = wordlist.keys()
        return word_features

    def document_features(self,document):
        document_words = set(document)
        # print "document_words ",len(all_words)
        features = {}
        for word in all_words:
            features['contains(%s)' % word] = (word in word_features)
        return features


dataExtractor = DataExtractor()
dataExtractor.extractFeatures()

featureModel = FeatureModel()
word_features = featureModel.getAllWords()

print "ALL LENGTH ",len(all_words)
print " word_features ",len(word_features)
print "All Tweets ",len(tweets)

feature_set = nltk.classify.util.apply_features(featureModel.document_features, tweets)
print "Classifier ",len(feature_set)

cutoff = len(feature_set[1:10])*2/3
train = feature_set[:cutoff]
test = feature_set[cutoff:]


classifier = nltk.NaiveBayesClassifier.train(train)
print "Accuracy Dude ",(nltk.classify.accuracy(classifier,test))
print "Most Informative Features ",classifier.show_most_informative_features(10)