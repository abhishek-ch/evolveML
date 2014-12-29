__author__ = 'abhishekchoudhary'
# https://github.com/databricks/learning-spark/blob/master/src/python/MLlib.py

from pyspark import SparkContext, SparkConf
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.classification import NaiveBayes
from pyspark.mllib.classification import LogisticRegressionWithSGD
import re
from nltk.corpus import stopwords

if __name__ == "__main__":
    htf = HashingTF()

    stopset = set(stopwords.words('english'))
    def cleanLine(line):
        # remove retweets
        line = re.sub("(RT|via)((?:\\b\\W*@\\w+)+)", "", line)
        line = re.sub("@\\w+", "", line)
        line = re.sub("http\\w+", "", line)
        line = re.sub("U+[a-zA-Z0-9]{0,10}", "", line)
        line = re.sub("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", line)
        # string is more than 4 characters
        # dataframe = dataframe[dataframe.x.str.len() > 4]
        # replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
        line = re.sub('[^\w\s]', "", line)

        if line.lower() not in stopset:
            return line.lower()
        else:
            return ""



    def myFunc(s):
        # words = s.split(",")
        s = re.sub("\"", "", s)
        words = [s for s in s.split(",")]
        val = words[0]
        lbl = 0.0
        if val == 4 or val == "4":
            lbl = 0.0
        elif val == 0 or val == "0":
            lbl = 1.0

        # print "words[0] %s words[5] %s"%(words[0],words[5])
        cleanlbl = cleanLine(words[5])
        # print "cleanlblcleanlbl ",cleanlbl
        return LabeledPoint(lbl, htf.transform(cleanlbl.split(" ")))


    conf = (SparkConf().setMaster("yarn-client").setAppName("LongDataSet-Spark"))
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    sc = SparkContext()
    sparseList = sc.textFile("hdfs:///stats/training.1600000.processed.noemoticon.csv").map(myFunc)
    sparseList.cache()  # Cache data since Logistic Regression is an iterative algorithm.


    # for data in dataset:
    trainfeats, testfeats = sparseList.randomSplit([0.8, 0.2], 10)

    # print "Length1 %s , Length %s "%(trainfeats.count(),testfeats.count())

    # model = LogisticRegressionWithSGD.train(trainfeats)  #Accuracy Rate 0.633956586072
    model = NaiveBayes.train(trainfeats, 2.0)              #Accuracy Rate 0.759233827702

    # accuracy testing
    # NOTE : This will throw error with using YARN
    labelsAndPreds = testfeats.map(lambda p: (p.label, model.predict(p.features)))
    labelsAndPreds.cache()
    accuracy = labelsAndPreds.filter(lambda (x, y): x == y).count()
    print "Accuracy ", accuracy / float(testfeats.count())

    print "Modelling1 ", model.predict(htf.transform("it rains a lot in london".split(" ")))
    print "Modelling2 ", model.predict(htf.transform("my life is so good , I am loving every moment of it".split(" ")))
    print "Modelling3 ", model.predict(
        htf.transform("you are a very bad and terrorism and evil killing children".split(" ")))
    print "Modelling4 ", model.predict(
        htf.transform("that was so wrong he did not forget about the bullshit".split(" ")))
    print "Modelling5 ", model.predict(htf.transform("Unfortunately the code isn\'t open source".split(" ")))

    sc.stop()