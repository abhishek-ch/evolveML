__author__ = 'achoudhary'

from bs4 import BeautifulSoup as bs
import os, sys, logging, string, glob
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row
import re
import nltk
from pyspark.sql.functions import *
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import IntegerType

#https://spark.apache.org/docs/1.4.1/sql-programming-guide.html - Dataframe
#https://spark.apache.org/docs/latest/api/python/pyspark.sql.html

#http://www.slideshare.net/BenjaminBengfort/fast-data-analytics-with-spark-and-python
#for brodcasting stop words - 54
#﻿spark.driver.extraClassPath	/home/cloudera/spark-1.4.1-bin-cdh4/lib/spark-csv_2.11-1.1.0.jar:/home/cloudera/spark-1.4.1-bin-cdh4/lib/commons-csv-1.1.jar
##above for adding spark-csv lib


#adding pipeline ML
#https://github.com/apache/spark/blob/master/examples/src/main/python/ml/cross_validator.py
#http://spark.apache.org/docs/latest/ml-guide.html


def parse_page(page, urlid):

    """ parameters:
            - in_file: file to read raw_data from
            - url_id: id of each page from file_name """


    """
    Arrange a way to to build the following as
    Spark Dataframe and then join the same with training csv dataframe

    Convert RDD of doc to List or tuple to convert the same to toDF
    RDD.toDF
                """
    doc = {}
    try:
        soup = bs(page,'html.parser')

        doc = {
                "id": urlid,
                "text":parse_text(soup),
                "title":parse_title(soup),
                "links":parse_links(soup),
                "images":parse_images(soup),
               }
    except Exception:
        print('Error')

    return doc


def parse_pageDataframe(page, urlid):
    output = Row(id = 'NA',text = 'NA',title='NA',links =0,images=0)
    
    try:
        soup = bs(page,'html.parser')
        output = Row(id = urlid,text = parse_text(soup),title=parse_title(soup),links =parse_links(soup),images=parse_images(soup))

    except Exception:
        print('Error')

    return output


def parse_text(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - textdata: a list of parsed text output by looping over html paragraph tags
        note:
            - could soup.get_text() instead but the output is more noisy """
    textdata = ['']

    for text in soup.find_all('p'):
        try:
            textdata.append(text.text.encode('ascii','ignore').strip())
        except Exception:
            continue

    text =  ' '.join(textdata)

    return cleanLine(text)



def cleanLine(line):
    
    # remove retweets
    line = re.sub("(RT|via)((?:\\b\\W*@\\w+)+)", "", line)
    line = re.sub("@\\w+", "", line)
    line = re.sub(r'[^\w\s]','',line)
    line = re.sub("http\\w+", "", line)
    line = re.sub("U+[a-zA-Z0-9]{0,10}", "", line)
    line = re.sub("[^(a-zA-Z0-9!@#$%&*(_) ]+", "", line)
    # string is more than 4 characters
    # dataframe = dataframe[dataframe.x.str.len() > 4]
    # replace all punctuation and ideally we will be expecting the hashtag is almost all the rows
    line = re.sub('[^\w\s]', "", line)
    
    # this is the extra work to handle

    #Replace occurance of all stop words to improvise and remove useless words
    line = ' '.join(c for c in line.split(" ") if not c in stopwords.value and len(c) > 2)
     
 

    return line.lower().strip()


def parse_title(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - title: parsed title """

    title = ['']

    try:
        title.append(soup.title.string.encode('ascii','ignore').strip())
    except Exception:
        return title

    return ' '.join(title)

def parse_links(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - linkdata: a list of parsed links by looping over html link tags
        note:
            - some bad links in here, this could use more processing """

    linkdata = ['']

    for link in soup.find_all('a'):
        try:
            linkdata.append(str(link.get('href').encode('ascii','ignore')))
        except Exception:
            continue

    return len(linkdata)


def parse_images(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - imagesdata: a list of parsed image names by looping over html img tags """
    imagesdata = ['']

    for image in soup.findAll("img"):
        try:
            imagesdata.append("%(src)s"%image)
        except Exception:
            continue

    return len(imagesdata)


def readContents(content):

    fileName = content[0]
    text = content[1]

    file = fileName.split("/")
    #print 'Each File Name {} f {}'.format(fileName,file[-1])
    #returns only the file name as ID
    if file[-1]+'\n' not in allValues.value:
    	return parse_pageDataframe(text,file[-1])

def getCleanedRDD(fileName,columns,htmldf):
    traindf = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load(fileName)
    joindf = htmldf.join(traindf, htmldf.id == traindf.file,'inner')

    tointfunc = UserDefinedFunction(lambda x: x,IntegerType())
    finaldf = joindf.withColumn("label",tointfunc(joindf['sponsored'])).select(columns)
    return finaldf


def main(args):

    outputDir = '/home/cloudera/Documents/output'
    dir = 'file:///home/cloudera/Documents/5'

    eachFile = dir.split("/")
    jsonFile = eachFile[-1]

    textFiles = sc.wholeTextFiles(dir).map(readContents)

    htmldf = sqlContext.createDataFrame(textFiles)
    htmldf.cache()
    #print dataframeText.show()
    '''
    traindf = sqlContext.read.format('com.databricks.spark.csv').options(header='true').load('/home/cloudera/Documents/train.csv')
    joindf = htmldf.join(traindf, htmldf.id == traindf.file,'inner')

    '''
    '''
    convert a Column to Integer type using UDF
    '''
    '''
    tointfunc = UserDefinedFunction(lambda x: x,IntegerType())
    finaldf = joindf.withColumn("label",tointfunc(joindf['sponsored'])).select("id","images","links","text","label")
    '''
    traindf = getCleanedRDD('/home/cloudera/Documents/train.csv',["id","images","links","text","label"],htmldf)
    print traindf.show()

    print '--------------------------------'
    testdf = getCleanedRDD('/home/cloudera/Documents/sampleSubmission.csv',["id","images","links","text","label"],htmldf)
    print testdf.show()

    #check bug
    #https://github.com/databricks/spark-csv/issues/64
    #finaldf.write.parquet(os.path.join(outputDir,"main_16.parquet"),'append')


if __name__ == "__main__":

   conf = (SparkConf().setMaster("local[2]").setAppName("Process HTML").set("spark.executor.memory", "4g"))
   sc = SparkContext(conf=conf)
   sqlContext = SQLContext(sc)
   stopwords = set(nltk.corpus.stopwords.words('english'))
   stopwords = sc.broadcast(stopwords)
   file = open('/home/cloudera/Documents/empty.txt','r')
   allValues = file.readlines()
   allValues = sc.broadcast(allValues)
   
   file.close()
   main(sys.argv)