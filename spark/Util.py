__author__ = 'abhishekchoudhary'

import pandas as pd


class Ready(object):
    def cleanDataFrame(self, dataframe):
        # remove retweets
        dataframe = dataframe.replace("(RT|via)((?:\\b\\W*@\\w+)+)", "", regex=True)
        dataframe = dataframe.replace("@\\w+", "", regex=True)
        dataframe = dataframe.replace("http\\w+", "", regex=True)
        dataframe = dataframe.replace("U+[a-zA-Z0-9]{0,10}", "", regex=True)
        dataframe = dataframe.replace("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", regex=True)
        # string is more than 4 characters
        # dataframe = dataframe[dataframe.x.str.len() > 4]
        # replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
        dataframe = dataframe.replace('[^\w\s]', "", regex=True)
        # convert all string to lower case
        dataframe.x = [x.lower().strip() for x in dataframe.tweets]

        # f = open('stop-words.txt', 'r')
        # stop_words = f.readlines()

        # if I don't believe in bigrams , I can use stop words to filter out values
        # for index, row in dataframe.iterrows():
        # if row in stop_words:
        # dataframe.loc[index]

        # need to work on stop words if any only if using unigram
        return dataframe


    def cleanLine(self, line):
        # remove retweets
        line = line.replace("(RT|via)((?:\\b\\W*@\\w+)+)", "", regex=True)
        line = line.replace("@\\w+", "", regex=True)
        line = line.replace("http\\w+", "", regex=True)
        line = line.replace("U+[a-zA-Z0-9]{0,10}", "", regex=True)
        line = line.replace("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", regex=True)
        # string is more than 4 characters
        # dataframe = dataframe[dataframe.x.str.len() > 4]
        # replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
        line = line.replace('[^\w\s]', "", regex=True)
        # convert all string to lower case
        # dataframe.x = [x.lower().strip() for x in dataframe.x]

        # f = open('stop-words.txt', 'r')
        # stop_words = f.readlines()

        # if I don't believe in bigrams , I can use stop words to filter out values
        # for index, row in dataframe.iterrows():
        # if row in stop_words:
        # dataframe.loc[index]

        # need to work on stop words if any only if using unigram
        return line.lower()


    def readFile(self, filepath, header=["tweets"]):
        # bigramData = sc.textFile(contentFile).cache()
        print(filepath)
        return pd.read_csv(filepath, names=header, header=-1)
