__author__ = 'abhishekchoudhary'
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os, string, re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
from sklearn import decomposition, pipeline, metrics, grid_search
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.decomposition import RandomizedPCA

# The following 3 functions have been taken from Ben Hamner's github repository
# https://github.com/benhamner/Metrics
def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings


def quadratic_weighted_kappa(y, y_pred):
    """
    Calculates the quadratic weighted kappa
    axquadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = y
    rater_b = y_pred
    min_rating = None
    max_rating = None
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert (len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return (1.0 - numerator / denominator)

# define stemmer
stemmer = PorterStemmer()
cachedStopWords = stopwords.words("english")


def tokenize(data):
    _cached_ = []
    data = BeautifulSoup(data).get_text()
    for word in data.split():
        word = word.lower()
        word = stemmer.stem(word)
        if word not in cachedStopWords and word not in string.punctuation and len(word) > 3:
            word = re.sub("http\\w+", "", word)
            word = re.sub("U+[a-zA-Z0-9]{0,10}", "", word)
            word = re.sub("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", word)
            #print(word)
            _cached_.append(word)

    return _cached_


def tokenize1(data):
    return data.split()


'''
Grid Search API
#http://isaacslavitt.com/2014/10/24/spdc-lightning-talk/
'''


def gridsearchWithData(pipeline, search_param, verbose_v=10, iid_v=True, cv_v=2):
    # Kappa Scorer
    kappa_scorer = metrics.make_scorer(quadratic_weighted_kappa, greater_is_better=True)

    # Initialize Grid Search Model
    model = grid_search.GridSearchCV(estimator=pipeline, param_grid=search_param, scoring=kappa_scorer,
                                     verbose=verbose_v, n_jobs=-1, iid=iid_v, refit=True, cv=cv_v)
    #model = grid_search.GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=2)

    # Fit Grid Search Model
    model.fit(traindata, train.median_relevance.values)
    print("Best score: %0.3f" % model.best_score_)
    print("Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(search_param.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    return model.best_estimator_


if __name__ == '__main__':
    BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..', 'crowdflower'))
    # Use Pandas to read in the training and test data
    train = pd.read_csv(BASE_DATA_PATH + "/train.csv").fillna("")
    test = pd.read_csv(BASE_DATA_PATH + "/test.csv").fillna("")

    # we dont need ID columns
    idx = test.id.values.astype(int)
    # drop the column
    train = train.drop('id', axis=1)
    test = test.drop('id', axis=1)

    print("LENGTH ", len(train.median_relevance))
    #train.product_description = list(map(lambda row: BeautifulSoup(row).get_text(), train.product_description))
    traindata = list(
        train.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1))
    testdata = list(
        test.apply(lambda x: '%s %s %s' % (x['query'], x['product_title'], x['product_description']), axis=1))

    count_vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize)
    tfidf = TfidfTransformer(norm="l2", smooth_idf=False, use_idf=False)

    #testdata = count_vectorizer.fit_transform(testdata)
    #testdata = tfidf.transform(testdata)




    ##########################################################NaiveBayes GridCV################################

    #http://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
    #http://w3facility.org/question/gridsearch-for-multilabel-onevsrestclassifier/
    pipeline_NaiveBayes = Pipeline([
        ('vec', CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize, min_df=1)),
        ('tfidf', TfidfTransformer(norm="l2", smooth_idf=False, use_idf=True)),
        ('classifier', MultinomialNB()
        )])



    #http://stackoverflow.com/questions/26569478/performing-grid-search-on-sklearn-naive-bayes-multinomialnb-on-multi-core-machin
    #GridSearchCV MultinomilaNB
    parameters_NaiveBayes = {
        'vec__max_features': (None, 200, 2000),
        'tfidf__use_idf': (True, False),
        'tfidf__norm': ('l1', 'l2'),
        'classifier__alpha': (1, 0.1, 0.001)
    }

    ######################################################################################

    #clf.fit(data, np.asarray(train.median_relevance))
    # Create a parameter grid to search for best parameters for everything in the pipeline
    param_grid = {'svd__n_components': [120, 140],
                  'svm__C': [1.0, 10]}


    ##############################################################ONeVSOne/Rest##############################
    #why probality = True (Sparse matrix)
    #http://stackoverflow.com/questions/27365020/scikit-learn-0-15-2-onevsrestclassifier-not-works-due-to-predict-proba-not-ava
    #classifier pipeline
    clf_pipeline = OneVsOneClassifier(
        Pipeline([
            ('reduce_dim', RandomizedPCA()),
            ('clf', SVC())
        ]
        ))

    #OneVsRest classfiier
    #http://stackoverflow.com/questions/12632992/gridsearch-for-an-estimator-inside-a-onevsrestclassifier
    parameters_OVR = {
        "classifier__estimator__clf__C": [1, 2, 4, 8],
        "classifier__estimator__clf__kernel": ["linear", "poly", "rbf"],
        "classifier__estimator__clf__degree": [1, 2, 3, 4]
    }
    #######################################################################################################

    ###########################################AdaBoost GridCV###############################################

    #Adaboost Classifier
    #https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/ensemble/tests/test_weight_boosting.py
    #http://codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn



    pipeline_Boost = Pipeline([
        ('vec', CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize, min_df=1)),
        ('tfidf', TfidfTransformer(norm="l2", smooth_idf=False, use_idf=True)),
        ('clf', AdaBoostClassifier(base_estimator=DecisionTreeClassifier())
        )])

    #n_estimators
    parameters_boost = {'clf__n_estimators': (1, 2),
                        'clf__algorithm': ('SAMME', 'SAMME.R'),
                        'clf__base_estimator__max_depth': (1, 2)
                        }

    #####################################################################################################


    best_model = gridsearchWithData(pipeline_Boost, parameters_boost)
    print(best_model)


    # Fit model with best parameters optimized for quadratic_weighted_kappa
    best_model.fit(traindata, train.median_relevance.values)
    preds = best_model.predict(testdata)

    # Create your first submission file
    submission = pd.DataFrame({"id": idx, "prediction": preds})
    submission.to_csv("output_Boost.csv", index=False)
