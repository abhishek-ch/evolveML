from mrjob.job import MRJob

#https://dataiap.github.io/dataiap/day5/mapreduce

import os


class MRWordFrequencyCount(MRJob):
    def mapper(self, _, line):
        yield "chars", len(line)
        yield "words", len(line.split())
        yield "lines", 1

    def reducer(self, key, values):
        yield key, sum(values)


DIRECTORY = '/Users/abhishekchoudhary/Downloads/aclImdb/test/try'


class MovieReview(MRJob):

    def mapper_init(self):
        self.lst = []
        for fname in os.listdir(DIRECTORY):
            file = open(os.path.join(DIRECTORY, fname))  # open a file
            for line in file:
                self.lst.append(line)
                yield "lines ", line

    def mapper(self, _, line):
        for val in self.lst:
            yield "lines ",val



if __name__ == '__main__':
    MovieReview.run()