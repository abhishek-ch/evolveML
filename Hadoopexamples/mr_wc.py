__author__ = 'abhishekchoudhary'
import os
from mrjob.protocol import JSONValueProtocol
from mrjob.job import MRJob
import get_terms

DIRECTORY = "/Users/abhishekchoudhary/Downloads/aclImdb/test/try/"


import re

STOPWORDS = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])

def get_terms(s):
    s = s.lower()
    terms = s.split()
    terms = map(lambda term: term.replace("'s",'').replace("'", '').replace(".", "").replace(",", ""), terms)
    terms = filter(lambda term: len(term) > 3, terms)
    terms = filter(lambda term: term not in STOPWORDS, terms)
    terms = filter(lambda term: re.match("^[a-zA-Z]+[a-zA-Z\'\-]*[a-zA-Z]+$", term), terms)
    return terms

class MRTFIDFBySender(MRJob):
    INPUT_PROTOCOL = JSONValueProtocol
    OUTPUT_PROTOCOL = JSONValueProtocol

    def mapper(self, key, email):
        for term in get_terms(email['text']):
            yield {'term': term, 'sender': email['sender']}, 1

    def reducer_init(self):
        self.idfs = {}
        for fname in os.listdir(DIRECTORY): # look through file names in the directory
            file = open(os.path.join(DIRECTORY, fname)) # open a file
            for line in file: # read each line in json file
                term_idf = JSONValueProtocol.read(line)[1] # parse the line as a JSON object
                self.idfs[term_idf['term']] = term_idf['idf']

    def reducer(self, term_sender, howmany):
        tfidf = sum(howmany) * self.idfs[term_sender['term']]
        yield None, {'term_sender': term_sender, 'tfidf': tfidf}


if __name__ == '__main__':
    MRTFIDFBySender.run()