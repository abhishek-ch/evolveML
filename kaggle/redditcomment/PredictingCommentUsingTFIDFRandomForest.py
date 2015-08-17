__author__ = 'achoudhary'
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


conn = sqlite3.connect('../input/database.sqlite')
cur = conn.cursor()

sql_cmd = "SELECT * FROM May2015  \
 \
limit 20000;"


df = pd.read_sql(sql_cmd,conn)

# Python reads as ascii so some characters throw errors if not converted
body_utf8 = [codecs.encode(i,'utf-8') for i in df.body]
df.body = pd.Series(body_utf8)

print(df.axes[1])

# Get length of comment and add to dataframe
df['body_length']=[len(i) for i in df.body]

# Convert categorical variable to 0 or 1
df['target']=[int(i=='moderator') for i in df['distinguished']]


# Tokenise
count_vect = CountVectorizer()
X_train = count_vect.fit_transform(df['body'])


# Transform to tf idf to extract useful features
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train)

# Split training and testing data
x_train, x_test, y_train, y_test = train_test_split(X_train_tfidf,df['target'],test_size=0.3)


# Random Forest model
clf = RandomForestClassifier(n_estimators=100,max_depth=300).fit(x_train,y_train)

# model scores
print("train F1 score: %f" %(f1_score(y_train,clf.predict(x_train))))
print("test F1 score: %f" %(f1_score(y_test,clf.predict(x_test))))
print("test recall score: %f" %(recall_score(y_test,clf.predict(x_test))))
print("test precision score: %f" %(precision_score(y_test,clf.predict(x_test))))
