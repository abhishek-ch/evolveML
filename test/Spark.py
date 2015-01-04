__author__ = 'abhishekchoudhary'

import os
import pandas as pd
from pyspark.mllib.feature import HashingTF, IDF, Vectors
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.regression import LabeledPoint
from pyspark import SparkContext, SparkConf
from nltk.corpus import stopwords
import string

def parseCSV(line):
    print "-----------------------------------------"
    values = [s for s in line.strip().split(",")]
    print "1 ====== %s AND 2 =>>>>>%s" % (values[0], values[3])


# conf = (SparkConf().setMaster("yarn-client").setAppName("My Long Data Set").set("spark.executor.memory", "1g"))
# sc = SparkContext("local[8]", appName="Test CSV")
# csv = sc.textFile("hdfs:///stats/testdata.manual.2009.06.14.csv").map(parseCSV)
# print("================>>>>>>>>>>>  ",csv.collect())
# # csv = csv.map(parseCSV)
# sc.stop()


Str = "Abhishek is the best don't guy I ever known for but thats disgu ever known for everthing ever be"
lis = ["is","don't","ever"]
str = ' '.join(c for c in Str.split(" ") if not c in lis)

print str

s = "asjo,fdjk;djaso,oio!kod.kjods;dkps"
print s.translate(None, ",!.;")