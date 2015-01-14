__author__ = 'achoudhary'

from collections import namedtuple

MovieData = namedtuple("movie","PhraseId SentenceId Phrase Sentiment")

MovieSingleData = namedtuple("movie",["PhraseId", "SentenceId","Phrase","Sentiment"])
