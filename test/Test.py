__author__ = 'achoudhary'
import pandas as pd
import os
import re

def readFile(filepath,header):
    # bigramData = sc.textFile(contentFile).cache()
    return pd.read_csv(filepath, names=[header],
                       header=0)


BASE_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data'))
# print BASE_DATA_PATH
stopwords = readFile(BASE_DATA_PATH+"/stop-words.txt","stop").stop.tolist()
# print stopwords
big_regex = re.compile(r'\b%s\b' % r'\b|\b'.join(map(re.escape, stopwords)),re.IGNORECASE)


val = ["Abhishek is the Some about life act of Words","Nothing is gng to change my baby world is enough"]



for words in val:
    #replace the string with stop-words which mostly make no sense
    the_message = big_regex.sub("",words)
    # the_message = re.sub(r'\s{2,100}',' ',the_message)
    #remove unnecessary white-space in-between string
    the_message = " ".join(the_message.split())
    print the_message.strip()

