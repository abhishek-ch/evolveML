__author__ = 'achoudhary'
#http://www.nltk.org/book/ch06.html
#http://www.tweenator.com/index.php?page_id=13
#http://help.sentiment140.com/for-students/

# import sys
# sys.path.append('/usr/local/lib/python2.7/site-packages/')
import pandas as pd
import nltk
import random


df = pd.read_csv("test.csv",header=0)
print df



def cleanDataFrame(dataframe):
        # remove retweets
        dataframe = dataframe.replace("(RT|via)((?:\\b\\W*@\\w+)+)", "", regex=True)
        # dataframe = dataframe.replace("@\\w+", "", regex=True)
        # dataframe = dataframe.replace("http\\w+", "", regex=True)
        dataframe = dataframe.replace("U+[a-zA-Z0-9]{0,10}", "", regex=True)
        dataframe = dataframe.replace("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", regex=True)
        # string is more than 4 characters
        # dataframe = dataframe[dataframe.x.str.len() > 4]
        # replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
        dataframe = dataframe.replace('[^\w\s]', "", regex=True)
        #convert all string to lower case
        dataframe.bond = [x.lower().strip() for x in dataframe.bond]

        f = open('stop-words.txt', 'r')
        stop_words = f.readlines()

        #if I don't believe in bigrams , I can use stop words to filter out values
        # for index, row in dataframe.iterrows():
        #     if row in stop_words:
        #         dataframe.loc[index]

        #need to work on stop words if any only if using unigram
        return dataframe

df = cleanDataFrame(df)

def readUnigrams():
    file = "/Users/abhishekchoudhary/Work/python/evolveML/py/post_neg2.txt"
    # bigramData = sc.textFile(contentFile).cache()
    return pd.read_csv(file, names=['term', 'sentimentScore', 'numPositive', 'numNegative'], sep='\t',
                       header=None)


unidf = readUnigrams()
print unidf.head()
# create random index
unidf = unidf.ix[random.sample(unidf.index, 2000)]
unidf = unidf.replace("@", "", regex=True)
unidf = unidf.replace("#", "", regex=True)
unidf = unidf.replace("http\\w+", "", regex=True)
all_words = unidf.term.tolist()
feature = []
positive = []
negative = []
all_words = unidf.term.tolist()
for index, row in unidf.iterrows():
    try:
        val = row['term']
        pos = row['numPositive']
        neg = row['numNegative']
        if val.startswith("http\\w+"):
            unidf.drop(index)
        else:
            if pos >= neg:
                positive.append(val)
            else:
                negative.append(val)


        feature.append([positive,'positive'])
        feature.append([negative,'negative'])
        # all_words = list(set(all_words))

    except AttributeError:
            unidf.drop(index)
    # print "valvalvalvalvalvalvalval",val
# print "------------------->> ",feature

def document_features(document):
    document_words = set(document)
    # print "document_words ",len(all_words)
    features = {}
    for word in all_words:
        features['contains(%s)' % word] = (word in document_words)
    return features

training_set = nltk.classify.util.apply_features(document_features, feature)
# featuresets = [(document_features(d), c) for (d,c) in feature]
# train_set, test_set = training_set[50:], training_set[51:100]
# print test_set
classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("=======>>> ",nltk.classify.accuracy(classifier, test_set))

testTweet = 'I feel bad for whoever has to clean up that mess'
print "YAHAHHAHAH ",classifier.classify(document_features(testTweet.split()))

def extract_features(document):
    features = {}
    for word in all_words:
        # print "Word ",word
        word = str(word)
        features['contains(%s)' % word] = (word in document)
    return features

# fe = extract_features('Disappointing day. Attended a car boot sale to raise some funds for the sanctuary, made a total of 88p after the entry fee - sigh')
# print "==================>",fe

#extract tweet list
tweets = []
for line in df.bond.tolist():
    value = line.split()
    tweets.append(value)


# Extract feature vector for all tweets in one shote
#training_set = nltk.classify.util.apply_features(extract_features, tweets)

# NBClassifier = nltk.NaiveBayesClassifier.train(feature)
# # print informative features about the classifier
# print NBClassifier.show_most_informative_features(10)