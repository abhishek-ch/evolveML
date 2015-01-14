__author__ = 'achoudhary'

import os
import csv
from MovieRecord import MovieSingleData

class DataReader(object):
    #http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-python-parameters
    #https://github.com/rafacarrascosa/samr/blob/develop/samr/corpus.py
    def readFilelinebyline(self,filepath):
        if os.path.exists(filepath):
            print 'Awesome'
            #read the file contents seperated by tab with ignoring the first row
            #using namedtuple here to arrange data one after
            iter = csv.reader(open(filepath,'r'),delimiter="\t")
            row = next(iter) #ignore first row

            for row in iter:
                #print 'length %s row %s '%(len(row),row[0])
                #yield MovieData(*row) #* passes the value as list
                yield MovieSingleData(row[0],row[1],row[2],row[3])

        else:
            print 'Sorry, but file path is invalid...'




if __name__ == '__main__':
    reader = DataReader()
    cached = []
    cached.extend(reader.readFilelinebyline('C:\\Users\\achoudhary\\Downloads\\train.tsv\\train.tsv'))