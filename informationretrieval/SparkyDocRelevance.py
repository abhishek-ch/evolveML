__author__ = 'abhishekchoudhary'

from bs4 import BeautifulSoup
import urllib
import re
from pyspark import SparkContext, SparkConf
import string
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
import graphlab
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

#http://ampcamp.berkeley.edu/big-data-mini-course/movie-recommendation-with-mllib.html
#graphlab.canvas.set_target('browser')

if __name__ == "__main__":
    hashingTF = HashingTF()
    htmlparser = re.compile("[0123456789+-.,!@#$%^&*();\/|<>#\"']+")
    space = re.compile("\\s+")
    url = re.compile("http:\/\/(.*?)/")
    tfidf_vectorizer = TfidfVectorizer()

    def visible(element):
        if element.parent.name in ['style', 'script', '[document]', 'head', 'title', 'img', 'i', 'a']:
            return False
        elif element == '\n':
            return False
        elif htmlparser.match(element):
            return False
        elif space.match(element):
            return False
        elif element == '':
            return False
        return True



    def readAllDocHDFS(document):
        try:
            # document = document.decode('utf-8').encode('utf-8')
            soup = BeautifulSoup(document)
            texts = soup.findAll(text=True)
            visible_texts = filter(visible, texts)
            all_values = ''.join(visible_texts)

            all_values = re.sub("http:\/\/(.*?)/", "", all_values)
            all_values = re.sub('\s+',' ', all_values)
            all_values = all_values.replace("\n"," ")
            return all_values
        except AttributeError, e:
            print()

    def calculateNormalTFIDF(document,val):
        print("1st ",document," 2nd ",val)
        tfidf_mat_train = tfidf_vectorizer.fit_transform(val)
        print(tfidf_mat_train)
        # simple_cos_sim = cosine_similarity(tfidf_mat_train[len(document) - 1], tfidf_mat_train)
        #
        # cosine_similarities = linear_kernel(tfidf_mat_train[len(document) - 1], tfidf_mat_train).flatten()
        # related_docs_indices = cosine_similarities.argsort()[:-5:-1]
        # print(related_docs_indices)
        # print(cosine_similarities[related_docs_indices])


    conf = (SparkConf().setMaster("local").setAppName("Trial"))
    sc = SparkContext(conf=conf)

    sparseList = sc.textFile("file:///Users/abhishekchoudhary/Work/python/raw2/*").map(readAllDocHDFS).reduce(calculateNormalTFIDF)
    sparseList.cache()
    # calculateNormalTFIDF(sparseList.collect())
    '''
    # sparseList = sc.textFile("file:///Users/abhishekchoudhary/Work/python/data/bank/banksample/*").map(readAllDocHDFS)
    # print(sparseList.collect())

    tf = hashingTF.transform(sparseList)
    # ... continue from the previous example
    tf.cache()

    idf = IDF().fit(tf)

    tfidf = idf.transform(tf)
    tfidf.cache()
    '''
    #graph of Frane
    #sf = graphlab.SFrame.from_rdd(rdd.pop(1))
    #sf.show()

    #sc.stop()