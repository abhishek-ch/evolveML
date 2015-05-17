__author__ = 'abhishekchoudhary'

import sys

from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithSGD


def parsePoint(line):
    """
    Parse a line of text into an MLlib LabeledPoint object.
    """
    values = [float(s) for s in line.split(' ')]
    if values[0] == -1:   # Convert -1 labels to 0 for MLlib
        values[0] = 0
    return LabeledPoint(values[0], values[1:])


if __name__ == "__main__":

    sc = SparkContext(appName="PythonLR")
    points = sc.textFile('file:///Users/abhishekchoudhary/Work/python/data/billionwords/train_v2.txt').map(parsePoint)
    iterations = int(20)
    model = LogisticRegressionWithSGD.train(points, iterations)
    print("Final weights: " + str(model.weights))
    print("Final intercept: " + str(model.intercept))
    sc.stop()