__author__ = 'achoudhary'
import pandas as pd
import os
import re
import numpy as np
import csv


with open('/Users/abhishekchoudhary/Downloads/trainingandtestdata/training.1600000.processed.noemoticon.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        statement = unicode(row[5], errors='replace')
        statement = statement.lower()
        statement = statement.replace("!@#$%^&*()[]{};:,./<>?\|`~-=_+", " ")
        print "row %s column %s " %(row[0],statement)



def readFile(filepath, header):
    # bigramData = sc.textFile(contentFile).cache()
    return pd.read_csv(filepath, names=[header],
                       header=0)


BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
# print BASE_DATA_PATH
stopwords = readFile(BASE_DATA_PATH + "/stop-words.txt", "stop").stop.tolist()
# print stopwords
big_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stopwords)), re.IGNORECASE)

val = ["Abhishek is the Some about life act of Words", "Nothing is gng to change my baby world is enough"]

for words in val:
    # replace the string with stop-words which mostly make no sense
    the_message = big_regex.sub("", words)
    # the_message = re.sub(r'\s{2,100}',' ',the_message)
    #remove unnecessary white-space in-between string
    the_message = " ".join(the_message.split())
    print the_message.strip()

d = []
d.append(("foo", 12))
d.append(("bar", 2))
d.append(("bob", 17))


print "DDDD ",d[1:]

np.random.shuffle(d)
for val,chal in d:
    print "VAL %s chal %s "%(val,chal)