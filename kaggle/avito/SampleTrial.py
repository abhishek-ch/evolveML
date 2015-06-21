__author__ = 'abc'
import os
from pyspark.sql import SQLContext
from pyspark.context import SparkContext, SparkConf
from pyspark.mllib.stat import Statistics
from pyspark.mllib.util import MLUtils
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD


def summarize(dataset):
    labels = dataset.map(lambda r: r.label)
    print("label average: %f" % labels.mean())
    features = dataset.map(lambda r: r.features)
    summary = Statistics.colStats(features)
    print("features average: %r" % summary.mean())


input = {'A': 1.0, 'B': 0.0}


def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """

    values = [s for s in line.split(',')]
    if values[0] == -1:  # Convert -1 labels to 0 for MLlib
        values[0] = 0

    #print values[0:-1]
    return LabeledPoint(input[values[-1]], values[0:-1])


if __name__ == '__main__':
    BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '../../../data', 'kaggle'))
    print(BASE_DATA_PATH)

    conf = (SparkConf().setMaster("local[2]").setAppName("Trial DF Set"))
    sc = SparkContext(conf=conf)

    # read data as CSV for Dataframe analysis
    # /Volumes/work/data/kaggle/ssi.csv


    # read data n0rmally
    '''

    sqlContext = SQLContext(sc)
    df = sqlContext.read.format('com.databricks.spark.csv').options(header='false').load(BASE_DATA_PATH + '/ssi.csv')
    # summarize(df)
    print df.show()

    #points = df.map(lambda row: LabeledPoint(input[row.C4],[float(row.C0),float(row.C1),float(row.C2),float(row.C3)]))

    values using Dataframe
    Final weights: [-137.221167143,12.555647803,53.629362055,109.314252441]
    Final intercept: 0.0
    '''

    points = sc.textFile(BASE_DATA_PATH+'/ssi.csv').map(parsePoint)
    model = LogisticRegressionWithSGD.train(points, 10)
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))


    '''
    Final weights: [-137.221167143,12.555647803,53.629362055,109.314252441]
    Final intercept: 0.0
    '''

    sc.stop()
