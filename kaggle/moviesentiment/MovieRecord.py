__author__ = 'achoudhary'

from collections import namedtuple

MovieData = namedtuple("movie","PhraseId SentenceId Phrase Sentiment")
TestMovieData = namedtuple("movie","PhraseId SentenceId Phrase")

MovieSingleData = namedtuple("movie",["PhraseId", "SentenceId","Phrase","Sentiment"])
