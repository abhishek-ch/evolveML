__author__ = 'abc'

import os
from pyspark.sql import SQLContext
from pyspark.context import SparkContext, SparkConf
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from pyspark.mllib.classification import SVMWithSGD, SVMModel,LogisticRegressionWithLBFGS

if __name__ == '__main__':
    BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '../../../data', 'kaggle'))
    print(BASE_DATA_PATH)

    conf = (SparkConf().setMaster("local[2]").setAppName("Testing MLLib With DataFrame SQL"))
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)
    # read the dataset
    df_test = sqlContext.read.format("com.databricks.spark.csv").options(delimiter=",").options(header="true").load(
        BASE_DATA_PATH + '/test.csv')

    training = df_test.map(lambda row: LabeledPoint(row.IsClick,
                                                    [float(row.SearchID), float(row.AdID), float(row.Position),
                                                     float(row.HistCTR), float(row.Price)]))

    (trainingData, testData) = training.randomSplit([0.7, 0.3])

    model = LinearRegressionWithSGD.train(training)



    # Build the model
    model1 = SVMWithSGD.train(trainingData, iterations=200)




    # Evaluate the model on training data


    model2 = RandomForest.trainClassifier(trainingData, numClasses=2,
                                         categoricalFeaturesInfo={},
                                         numTrees=3, featureSubsetStrategy="auto",
                                         impurity='gini', maxDepth=4, maxBins=32)




    # Build the model
    model3 = LogisticRegressionWithLBFGS.train(trainingData)



    model4 = GradientBoostedTrees.trainClassifier(trainingData,
        categoricalFeaturesInfo={}, numIterations=3)




        # Evaluate model on test instances and compute test error
    predictions = model1.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    print('Test Mean Squared Error Model1= ' + str(testMSE))

        # Evaluate model on test instances and compute test error
    predictions = model2.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    print('Test Mean Squared Error Model2= ' + str(testMSE))


          # Evaluate model on test instances and compute test error
    predictions = model3.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    print('Test Mean Squared Error Model3= ' + str(testMSE))

      # Evaluate model on test instances and compute test error
    predictions = model4.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
    print('Test Mean Squared Error Model4= ' + str(testMSE))

    print('{} {} {} - {} - {}'.format(
        model.predict([float(37653743), float(13558903), float(1), float(0.044174), float(1550)]),
        model1.predict([float(37653743), float(13558903), float(1), float(0.044174), float(1550)]),
        model2.predict([float(37653743), float(13558903), float(1), float(0.044174), float(1550)]),
        model3.predict([float(37653743), float(13558903), float(1), float(0.044174), float(1550)]),
        model4.predict([float(37653743), float(13558903), float(1), float(0.044174), float(1550)])

    ))

    print(
    '{} {} {} - {} - {}'.format(model.predict([float(8023880), float(2809658), float(1), float(0.009), float(9829)]),
                   model1.predict([float(8023880), float(2809658), float(1), float(0.009), float(9829)]),
                   model2.predict([float(8023880), float(2809658), float(1), float(0.009), float(9829)]),
                    model3.predict([float(8023880), float(2809658), float(1), float(0.009), float(9829)]),
                   model4.predict([float(8023880), float(2809658), float(1), float(0.009), float(9829)])
                      )
    )

    print('{} {} {} - {} - {}'.format(model.predict([float(123456), float(122344), float(7), float(0.0198689), float(20019)]),
                         model1.predict([float(123456), float(122344), float(7), float(0.0198689), float(20019)]),
                         model2.predict([float(123456), float(122344), float(7), float(0.0198689), float(20019)]),
                            model3.predict([float(123456), float(122344), float(7), float(0.0198689), float(20019)]),
                         model4.predict([float(123456), float(122344), float(7), float(0.0198689), float(20019)])
                         ))

    print('{} {} {} - {} - {}'.format(model.predict([float(64681007), float(28070889), float(1), float(0.008549), float(9090)]),
                         model1.predict([float(64681007), float(28070889), float(1), float(0.008549), float(9090)]),
                         model2.predict([float(64681007), float(28070889), float(1), float(0.008549), float(9090)]),
                            model3.predict([float(64681007), float(28070889), float(1), float(0.008549), float(9090)]),
                         model4.predict([float(64681007), float(28070889), float(1), float(0.008549), float(9090)])
                         ))
