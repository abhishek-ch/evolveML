__author__ = 'achoudhary'

from bs4 import BeautifulSoup as bs
import os, sys, logging, string, glob
import json
from pyspark import SparkConf, SparkContext
from pyspark.sql import SQLContext, Row

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
        soup = bs(page)

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
        soup = bs(page)
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

    return ' '.join(textdata)

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
    return parse_pageDataframe(text,fileName)

def main(args):

    outputDir = '/home/cloudera/Documents/output'
    dir = 'file:///home/cloudera/Documents/0'

    eachFile = dir.split("/")
    jsonFile = eachFile[-1]

    textFiles = sc.wholeTextFiles(dir).map(readContents)

    dataframeText = sqlContext.createDataFrame(textFiles)
    #print dataframeText.show()
    
    dataframeText.write.parquet(os.path.join(outputDir,"main_0.parquet"))

    '''
    print 'json file Name {}'.format(textFiles.take(1))
    out_file = os.path.join(outputDir, 'jsonFile_1.json')
    with open(out_file, mode='w') as feedsjson:
        for val in textFiles.collect():
            json.dump(val, feedsjson)
            feedsjson.write('\n')

    feedsjson.close()
    '''

if __name__ == "__main__":

   conf = (SparkConf().setMaster("local[2]").setAppName("Process HTML").set("spark.executor.memory", "6g"))
   sc = SparkContext(conf=conf)
   sqlContext = SQLContext(sc)
   main(sys.argv)