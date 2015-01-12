__author__ = 'abhishekchoudhary'

import os
import pandas as pd
from pyspark.mllib.feature import HashingTF, IDF, Vectors
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext,SparkConf
from nltk.corpus import stopwords
# from Util import Ready

#http://spark.apache.org/docs/latest/programming-guide.html




class TimePass(object):
    def parsingCSV(self,line):
        print "-----------------------------------------"
        values = [s for s in line.strip().split(",")]
        print "1 ====== %s AND 2 =>>>>>%s"% (values[0], values[3])



htf = HashingTF()
def myFunc(s):
    print("TESTING------------")
    words = s.split(",")
    return LabeledPoint(1.0, htf.transform(words[0]))

def myFunc(s):
    words = s.split(" ")
    print "length", len(words)

conf = (SparkConf().setMaster("local").setAppName("My Long Data Set").set("spark.executor.memory", "1g"))
conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
sc = SparkContext(conf=conf)
csv = sc.textFile("hdfs:///stats/test.csv").map(myFunc)
sc.stop()



# timepass = TimePass()
# test = csv.map(myFunc)

'''
mapValue = csv.map(lambda x: (x.split(","), x))
maplist = mapValue.collect()
maplist = maplist[len(maplist)-1][0]
print "========##################### ", maplist
'''
