#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
# plt.xlim(0.0, 1.0)
# plt.ylim(0.0, 1.0)
# plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
# plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
# plt.legend()
# plt.xlabel("bumpiness")
# plt.ylabel("grade")
# plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
# http://sebastianraschka.com/Articles/2014_ensemble_classifier.html
#NaiveBayes = 93.6

from sklearn.neighbors import KNeighborsClassifier
from time import time


t0 = time()

clf = KNeighborsClassifier(n_neighbors=3)
clf = clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"


t1 = time()
predict = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"

import numpy as np
import math
from sklearn.metrics import accuracy_score

print("Accuracy" ,accuracy_score(labels_test, predict))
# try:
#     prettyPicture(clf, features_test, labels_test)
# except NameError:
#     pass

from sklearn.ensemble import BaggingClassifier
bagging = BaggingClassifier(KNeighborsClassifier(),n_estimators=300, max_samples=0.4, max_features=2)
bagging.fit(features_train, labels_train)
print("Accuracy_BaggingClassifier %0.4f" % (accuracy_score(labels_test, bagging.predict(features_test))))
# bagging = BaggingClassifier(RandomForestClassifier(),n_estimators=300, max_samples=0.5, max_features=2)

# try:
#     prettyPicture(bagging, features_test, labels_test,"bagging.png")
# except NameError:
#     pass




from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf_adaboost = AdaBoostClassifier(RandomForestClassifier(max_depth=None,n_estimators=500),
                         algorithm="SAMME",n_estimators=200,random_state=2)
# scores = cross_val_score(clf_adaboost, features_train, labels_train)
# print("Adaboost Score Mean",scores.mean())
clf_adaboost = clf_adaboost.fit(features_train, labels_train)
print("Adaboost Decision Tree Predict %0.4f" % (accuracy_score(labels_test, clf_adaboost.predict(features_test))))
# try:
#     prettyPicture(clf_adaboost, features_test, labels_test,"clf_adaboost2.png")
# except NameError:
#     pass

# clf_random = RandomForestClassifier(n_estimators=100, max_depth=None,min_samples_split=1, random_state=0)
# scores = cross_val_score(clf_random, features_train, labels_train)
# print("mean ",scores.mean())


clf2 = RandomForestClassifier(n_estimators=200, max_depth=2,min_samples_split=1, random_state=1)
clf3 = GaussianNB()

print('5-fold cross validation:\n')


for clf_main, label in zip([clf, clf2, clf3], ['KNearestForest', 'Random Forest', 'naive Bayes']):
    scores = cross_validation.cross_val_score(clf_main, features_train, labels_train, cv=5, scoring='accuracy')    
    clf_main = clf_main.fit(features_train,labels_train)
    predict_main = clf_main.predict(features_test)
    print("Accuracy: %0.2f (+/- %0.2f) [%s] Prediction: %0.4f" % (scores.mean(), scores.std(), label,accuracy_score(labels_test, predict_main)))






