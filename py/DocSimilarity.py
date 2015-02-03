__author__ = 'achoudhary'
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'raw2'))
# print(BASE_DATA_PATH)
documents = [os.path.join(BASE_DATA_PATH, fn) for fn in next(os.walk(BASE_DATA_PATH))[2]]

test = "which after all was not very far out of myway, instead of striking straight back ransport riders and moment the sermon improved by degrees, till at length"

train_set = [test, documents]
documents.append(test)

tfidf_vectorizer = TfidfVectorizer()
tfidf_mat_train = tfidf_vectorizer.fit_transform(documents)
print(tfidf_mat_train)
simple_cos_sim = cosine_similarity(tfidf_mat_train[len(documents) - 1], tfidf_mat_train)
print("Simple ", simple_cos_sim)

cosine_similarities = linear_kernel(tfidf_mat_train[len(documents) - 1], tfidf_mat_train).flatten()
print(cosine_similarities)
#print("Cosine Similarity ",cosine_similarity(test,tfidf_mat_train))

related_docs_indices = cosine_similarities.argsort()[:-5:-1]
print(related_docs_indices)
print(cosine_similarities[related_docs_indices])