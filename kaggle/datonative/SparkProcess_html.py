__author__ = 'achoudhary'

from bs4 import BeautifulSoup as bs
import os, sys, logging, string, glob
import cssutils as cu
import json
from pyspark import SparkConf, SparkContext

def parse_page(page, urlid):
    """ parameters:
            - in_file: file to read raw_data from
            - url_id: id of each page from file_name """
    soup = bs(page)
    doc = {
            "id": urlid,
            "text":parse_text(soup),
            "title":parse_title(soup ),
            "links":parse_links(soup),
            "images":parse_images(soup),
           }
    json_array.append(doc)
    return doc

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

    return filter(None,textdata)

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

    return filter(None,title)

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

    return filter(None,linkdata)


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

    return filter(None,imagesdata)


def readContents(content):

    fileName = content[0]
    text = content[1]
    print 'Each File Name '.format(fileName)
    return parse_page(text,fileName)

def main(args):
    outputDir = '/home/cloudera/Documents/'
    dir = 'file:///home/cloudera/Documents/files'

    eachFile = dir.split("/")
    jsonFile = eachFile[-1]

    textFiles = sc.wholeTextFiles(dir).map(readContents)

    print 'json file Name {}'.format(jsonFile)
    out_file = os.path.join(outputDir, 'jsonFile_0.json')
    with open(out_file, mode='w') as feedsjson:
        for entry in json_array:
            json.dump(entry, feedsjson)
            feedsjson.write('\n')

    feedsjson.close()

if __name__ == "__main__":

   conf = (SparkConf().setMaster("local[2]").setAppName("Process HTML").set("spark.executor.memory", "2g"))
   sc = SparkContext(conf=conf)
   json_array = []
   main(sys.argv)