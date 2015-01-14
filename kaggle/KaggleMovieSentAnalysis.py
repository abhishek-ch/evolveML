__author__ = 'achoudhary'

import os
import csv
from MovieRecord import MovieSingleData,MovieData

class DataReader(object):
    #http://stackoverflow.com/questions/36901/what-does-double-star-and-star-do-for-python-parameters
    #https://github.com/rafacarrascosa/samr/blob/develop/samr/corpus.py
    def readFilelinebyline(self,DIRECTORY):
        filepath = os.path.join(DIRECTORY, 'train.tsv')
        if os.path.exists(filepath):
            print 'Awesome'
            #read the file contents seperated by tab with ignoring the first row
            #using namedtuple here to arrange data one after
            iter = csv.reader(open(filepath,'r'),delimiter="\t")
            row = next(iter) #ignore first row

            for row in iter:
                yield MovieData(*row)   #* passes the value as list
                #yield MovieSingleData(row[0],row[1],row[2],row[3])

        else:
            print 'Sorry, but file path is invalid...'

    def getData(self,datalist=[]):
        #get current project root -> one step up -> get data dir -> get the file
        BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..','..', 'data'))
        datalist.extend(self.readFilelinebyline(BASE_DATA_PATH))
        return datalist


    def getTrainTestData(self):
        reader = DataReader()
        data = list(reader.getData())
        return data


if __name__ == '__main__':
    reader = DataReader()
    data = reader.getTrainTestData()
    for val in data:
        print(val.Phrase)