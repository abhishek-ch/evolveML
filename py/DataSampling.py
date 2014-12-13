__author__ = 'achoudhary'

#http://www.laurentluce.com/posts/twitter-sentiment-analysis-using-python-and-nltk/

# data exploration in python shell
# http://ampcamp.berkeley.edu/big-data-mini-course/data-exploration-using-spark.html

from pyspark import SparkContext


import pprint
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages/')
import pandas as pd
import nltk
import time
import csv


class DataSampling(object):
    def loadcsv(self, filename):
        df = pd.read_csv(filename)
        print "=====> ",df.describe()
        return df


    def createDataframe(self,input):
        df = pd.DataFrame(input[1:],columns=['x'])
        print df.head()
        return df

    def cleanDataFrame(selfself, dataframe):
        # remove retweets
        dataframe = dataframe.replace("(RT|via)((?:\\b\\W*@\\w+)+)", "", regex=True)
        dataframe = dataframe.replace("@\\w+", "", regex=True)
        dataframe = dataframe.replace("http\\w+", "", regex=True)
        dataframe = dataframe.replace("U+[a-zA-Z0-9]{0,10}", "", regex=True)
        dataframe = dataframe.replace("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", regex=True)
        #string is more than 4 characters
        dataframe = dataframe[dataframe.x.str.len() > 4]
        #replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
        dataframe = dataframe.replace('[^\w\s]', "", regex=True)
        #convert all string to lower case
        dataframe.x = [x.lower().strip() for x in dataframe.x]
        return dataframe


class Frequency(object):
    def findWordFrequency(self, all_words):
        wordlist = nltk.FreqDist(all_words)
        print 'Printing ',wordlist.most_common(50)
        return wordlist.keys()


    def extractAllWords(self, df):
        all_words = []
        for index,row in df.iterrows():
            #split all the value to each specific words
            all_words.extend((''.join(row)).split())
       # print df.x
        return all_words


sc = SparkContext("yarn-client", "Data Sampling")
contentFile = "hdfs:///stats/foo5.csv"
logData = sc.textFile(contentFile).cache()
# print "LOGDTATATATATAT ",logData.collect()
sampling = DataSampling()
df = sampling.createDataframe(logData.take(logData.count()))
# df = sampling.loadcsv("/Users/abhishekchoudhary/Work/python/evolveML/foo5.csv")

df = sampling.cleanDataFrame(df)
# print df.head()

frequence = Frequency()
all_words = frequence.extractAllWords(df)
frequence.findWordFrequency(all_words)

