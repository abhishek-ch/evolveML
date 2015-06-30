__author__ = 'abc'
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel


def predict_proba(rf_model, data):
    '''
    This wrapper overcomes the "binary" nature of predictions in the native
    RandomForestModel.
    '''  # Collect the individual decision tree models by calling the underlying
    # Java model. These are returned as JavaArray defined by py4j.
    trees = rf_model._java_model.trees()
    ntrees = rf_model.numTrees()
    scores = DecisionTreeModel(trees[0]).predict(data.map(
        lambda row: [float(row.SearchID), float(row.AdID), float(row.Position), float(row.ObjectType),
                     float(row.HistCTR)]))

    # For each decision tree, apply its prediction to the entire dataset and
    # accumulate the results using 'zip'.
    for i in range(1, ntrees):
        dtm = DecisionTreeModel(trees[i])
        scores = scores.zip(dtm.predict(data.map(lambda row : [float(row.SearchID),float(row.AdID),float(row.Position),float(row.ObjectType),float(row.HistCTR)])))
        scores = scores.map(lambda x: x[0] + x[1])

    # Divide the accumulated scores over the number of trees
    return scores.map(lambda x: x / ntrees)
