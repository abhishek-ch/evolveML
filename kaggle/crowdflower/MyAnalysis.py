__author__ = 'abhishekchoudhary'
import pandas as pd

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
import os,string,re

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..', 'crowdflower'))
# Use Pandas to read in the training and test data
train = pd.read_csv(BASE_DATA_PATH+"/train.csv").fillna("")
test  = pd.read_csv(BASE_DATA_PATH+"/test.csv").fillna("")

# we dont need ID columns
idx = test.id.values.astype(int)
#drop the column
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

cachedStopWords = stopwords.words("english")
#define stemmer
stemmer = PorterStemmer()

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

print("LENGTH ",len(train.median_relevance))
#train.product_description = list(map(lambda row: BeautifulSoup(row).get_text(), train.product_description))
traindata = list(train.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
testdata = list(test.apply(lambda x:'%s %s %s' % (x['query'],x['product_title'], x['product_description']),axis=1))
#print(traindata[1:5])

count_vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize)
count = count_vectorizer.fit_transform(traindata)

tfidf = TfidfTransformer(norm="l2", smooth_idf=False, use_idf=False)
data = tfidf.fit_transform(count)

classfier = OneVsRestClassifier(LinearSVC())
classfier.fit(data, np.asarray(train.median_relevance))


count_test = count_vectorizer.transform(testdata)
tfidf_test = tfidf.transform(count_test)
predicted = classfier.predict(tfidf_test)

# Create your first submission file
submission = pd.DataFrame({"id": idx, "prediction": predicted})
submission.to_csv("output2.csv", index=False)