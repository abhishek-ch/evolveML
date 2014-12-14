__author__ = 'abhishekchoudhary'
import sys
import math

sys.path.append('/usr/local/lib/python2.7/site-packages/')
import pandas as pd


input = [u'x',
         u"RT @jvsk3: Banning is d way 2 hide once incompetency's &amp; BJP ban is 2 hide corruption which led 2 crime  https://t.co/VOR9EVll9U #DelhiSham ",
         u'RT @NoName4840:                  (      ) https://t.co/WrCrzRrhpM #IslamicState #IS #ISIS #ISIL #ISID #India  #DelhiShamedAgain',
         u'RT @DelhiDialogue: BJP-led SDMC. Another poor municipal corporation. Women bear the worst of their ineffectiveness.. #DelhiShamedAgain http ',
         u'RT @DelhiDialogue: CCTVs in buses and bus shelters. Make private taxis redundant by making public transport safe. #DelhiShamedAgain http:// ',
         u'"RT @DrunkVinodMehta: Cab Driver was everyone you by Court in earlier rape case. If Modi be PM after 2002 riots accuse, why can\'t he drive a Ca "',
         u'"RT @DrunkVinodMehta: NihalChand accused of Rape, eligible for Cabinet Minister. Cab Driver accused of rape, #Uber shld not hv employed him. "',
         u'RT @jvsk3: #DelhiShamedAgain If character certificate was not issued by Delhi police would accused cab driver got d job?  https://t.co/VOR9 ',
         u'Passport address verification is done by giving 500  to police. Do u think they will do driver verification diligently ? #DelhiShamedAgain',
         u"RT @DelhiDialogue: BJP-led NDMC why limit fiscal prudence on women's issues when it needs proactive governance. #DelhiShamedAgain http: "]

df = pd.DataFrame(input[1:], columns=['x'])
# print df.head()

f = open('stop-words.txt', 'r')
stop_words = f.readlines()


for index,row in df.iterrows():
    print "ABHISHKE",df.loc[index,'x']
# df = df[df.x.str.len() > 4]
# print df

import csv

bigrams = "/Users/abhishekchoudhary/Downloads/Sentiment140-Lexicon-v0.1/bigrams-pmilexicon copy.txt"

football = pd.read_csv(bigrams, names=['term', 'sentimentScore', 'numPositive', 'numNegative'], sep='\t', header=None)

i = 0
features = {}
for word in football.term.tolist():
    if any(word in s for s in input):
        features['typeof(%s)' % word] = ("Positive"
                if football['numPositive'][i] > football['numNegative'][i] else "Negative")
    else:
        features['typeof(%s)' % word]=("NoResponse")
    i += 1

# print  features


pos_tweets = [('I love this car', 'positive'),
              ('This view is amazing', 'positive'),
              ('I feel great this morning', 'positive'),
              ('I am so excited about the concert', 'positive'),
              ('He is my best friend', 'positive')]


neg_tweets = [('I do not like this car', 'negative'),
              ('This view is horrible', 'negative'),
              ('I feel tired this morning', 'negative'),
              ('I am not looking forward to the concert', 'negative'),
              ('He is my enemy', 'negative')]


tweets = []
for (words, sentiment) in pos_tweets + neg_tweets:
    words_filtered = [e.lower() for e in words.split() if len(e) >= 3]
    tweets.append((words_filtered, sentiment))

print "TWTWTWTWT ",tweets


#start extract_features
def extract_features(tweet):
    tweet_words = set(tweet)
    print "---------",tweet_words
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features

extract_features(tweets)

# for val in stop_words:
#     print val

def readUnigrams():
    file = "/Users/abhishekchoudhary/Work/python/evolveML/py/post_neg.txt"
    # bigramData = sc.textFile(contentFile).cache()
    return pd.read_csv(file, names=['term', 'sentimentScore', 'numPositive', 'numNegative'], sep='\t',
                       header=None)


unidf = readUnigrams()
print unidf.head()
unidf = unidf.replace("@", "", regex=True)
unidf = unidf.replace("#", "", regex=True)
unidf = unidf.replace("http\\w+", "", regex=True)
feature = []
positive = []
negative = []
for index, row in unidf.iterrows():
    try:
        val = row['term']
        pos = row['numPositive']
        neg = row['numNegative']
        if val.startswith("http\\w+"):
            unidf.drop(index)
        else:
            if pos > neg:
                positive.append(val)
            else:
                negative.append(val)

        feature.append((positive,'positive'))
        feature.append((negative,'negative'))

    except AttributeError:
            unidf.drop(index)
    # print "valvalvalvalvalvalvalval",val

# print feature




