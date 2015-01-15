__author__ = 'achoudhary'

import os
import csv
from MovieRecord import MovieSingleData, MovieData
import random
import nltk
import collections
import string
from nltk.corpus import stopwords
import re

class DataReader(object):
    def __init__(self):
        self.signature = ['Mr.', 'Mrs.', 'Dr.', 'Ms.', 'Miss.']
        self.stop_words = list(set(stopwords.words('english')))
        self.remove_words = ["'ve","'nt","'ll","n't",'...',"'re'"]
        self.pattern1 = re.compile("^[.-]+\w+[.-]+$")
        self.wordPattern = re.compile("^[\w\d]*[\-\'][\w\d]+$")

        self.most_significant_words = []
    # http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-python-parameters
    # https://github.com/rafacarrascosa/samr/blob/develop/samr/corpus.py
    def readFilelinebyline(self, DIRECTORY):
        filepath = os.path.join(DIRECTORY, 'train.tsv')
        if os.path.exists(filepath):
            print 'Awesome'
            #read the file contents seperated by tab with ignoring the first row
            #using namedtuple here to arrange data one after
            iter = csv.reader(open(filepath, 'r'), delimiter="\t")
            next(iter)  #ignore first row

            for row in iter:
                yield MovieData(*row)  #* passes the value as list
                #yield MovieSingleData(row[0],row[1],row[2],row[3])

        else:
            print 'Sorry, but file path is invalid...'

    def getData(self, datalist=[]):
        #get current project root -> one step up -> get data dir -> get the file
        BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
        datalist.extend(self.readFilelinebyline(BASE_DATA_PATH))

        #shuffle the entire dataset
        #random.shuffle(datalist)

        return datalist

    def cleanData(self, data=[]):
        specific_words_dict = collections.defaultdict(list)
        all_words = []
        if len(data) < 2:
            print 'Sorry but its too less data for ML'
        else:
            for line in data:
                phrase = line.Phrase
                sentiment = line.Sentiment
                i = 0
                for word in phrase.split():
                    #extract name and name honouring
                    if word in self.signature or word in string.punctuation:
                        continue
                    if word[0].upper() and i > 0:
                        continue
                    if word.lower() in self.stop_words or len(word) <= 2:
                        continue
                    if word.lower() in self.remove_words:
                        continue
                    if self.pattern1.match(word.lower()):
                        continue
                    all_words.append(word.lower())
                    specific_words_dict[sentiment].append(word.lower())
                    #For First Worst Check
                    i += 1


        return specific_words_dict,all_words

    #find the frequency of the all word list
    def wordFrequency(self, data=[]):
        if len(data) < 2:
            return ['Sorry but its too less data for ML']
        else:
            specific_words_dict,all_words = self.cleanData(data)
            word_features = nltk.FreqDist(all_words)
            return specific_words_dict,all_words,word_features.most_common(6000)





    def getTrainTestData(self):
        reader = DataReader()
        data = list(reader.getData())

        specific_words_dict,all_words,self.most_significant_words = self.wordFrequency(data)
        print(self.most_significant_words[1:200])
        print(len(self.most_significant_words))
        #self.wordFrequency(data)
        return data


if __name__ == '__main__':
    reader = DataReader()
    data = reader.getTrainTestData()
    #for val in data:
     #   print(val.Phrase)