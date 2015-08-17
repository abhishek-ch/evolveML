__author__ = 'abc'

import zipfile
import pandas as pd
import json
import math
import multiprocessing as mp
import os
import glob
import logging
import time
import sys
import glob
from bs4 import BeautifulSoup as bs


def parse_page(in_file, urlid):
    """ parameters:
            - in_file: file to read raw_data from
            - url_id: id of each page from file_name """
    # print len(in_file)
    #if len(in_file) < 2:
    #    return -1

    page = in_file
    soup = bs(page)
    doc = {
        "id": urlid,
        "text": parse_text(soup),
        "title": parse_title(soup),
        "links": parse_links(soup),
        "images": parse_images(soup),
    }

    yield doc


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
            textdata.append(text.text.encode('ascii', 'ignore').strip())
        except Exception:
            # print 'Error while parse_text {}'.format(text)
            continue
    # print textdata
    return filter(None, textdata)


def parse_title(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - title: parsed title """

    title = ['']

    try:
        title.append(soup.title.string.encode('ascii', 'ignore').strip("\s"))
    except Exception:
        # print 'Error while parse_title {}'.format(title)
        return title
    return filter(None, title)


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
            linkdata.append(str(link.get('href').encode('ascii', 'ignore')))
        except Exception:
            # print 'Error while parse_link {}'.format(link)
            continue

    return filter(None, linkdata)


def parse_images(soup):
    """ parameters:
            - soup: beautifulSoup4 parsed html page
        out:
            - imagesdata: a list of parsed image names by looping over html img tags """
    imagesdata = ['']

    for image in soup.findAll("img"):
        try:
            imagesdata.append("%(src)s" % image)
        except Exception:
            # print 'Error while parse_image {}'.format(image)
            continue

    return filter(None, imagesdata)


def readDataFromeArchieve(zipfile, filename):
    filenameDetails = filename.split("/")
    bucket = filenameDetails[-1].split('.zip')[0]
    urls = file.split("/")
    urlId = urls[-1].split('_')[0]
    yield bucket, urlId, zipfile.read(file)


def yield_data_from_zip(archive_path):
    archive = zipfile.ZipFile(archive_path, 'r')
    file_paths = zipfile.ZipFile.namelist(archive)
    for file_path in file_paths:
        urls = file_path.split("/")
        urlId = urls[-1].split('_')[0]

        data = archive.read(file_path)
        yield data, urlId


def readUpdate(args):
    inFolder = args[1]
    outputDirectory = args[2]
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    idx = 0

    for filename in glob.glob(os.path.join(inFolder, '*.zip')):
        json_array = []
        filenameDetails = filename.split("/")
        last_bucket = filenameDetails[-1].split('.zip')[0]
        for data, urlId in yield_data_from_zip(filename):
            idx += 1
            if idx % 10000 == 0:
                print "Processed {} HTML file name {} and file {}" .format(idx, file,urlId)
            #print 'urlId {}'.format( urlId)
            doc = parse_page(data, urlId)
            json_array.append(doc)

        out_file = os.path.join(outputDirectory, 'chunk_' + last_bucket + '.json')
        with open(out_file, mode='w') as feedsjson:
            for entry in json_array:
                for value in entry:
                    json.dump(value, feedsjson)
                    feedsjson.write('\n')

        feedsjson.close()


def read(args):
    inFolder = args[1]
    outputDirectory = args[2]

    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    json_array, last_bucket = [], str(0)

    # pool = mp.Pool(processes=2)

    for filename in glob.glob(os.path.join(inFolder, '*.zip')):
        print(filename)

        zf = zipfile.ZipFile(filename, 'r')
        filelist = zf.namelist()
        idx = 0
        for file in readDataFromeArchieve(zf, filename):

            filenameDetails = filename.split("/")
            bucket = filenameDetails[-1].split('.zip')[0]
            urls = file.split("/")
            urlId = urls[-1].split('_')[0]

            if idx % 1000 == 0:
                print "Processed %d HTML file name %s" % (idx, file)

            idx += 1

            if bucket != last_bucket:
                print 'SAVING BUCKET %s' % last_bucket
                out_file = os.path.join(outputDirectory, 'chunk' + last_bucket + '.json')
                with open(out_file, mode='w') as feedsjson:
                    for entry in json_array:
                        json.dump(entry, feedsjson)
                        feedsjson.write('\n')

                feedsjson.close()
                json_array = []
                last_bucket = bucket
            try:
                print 'len {} xid {} urlId {}'.format(len(json_array), idx, urlId)
                data = zf.read(file)
                doc = parse_page(data, urlId)
                # doc = pool.apply(parse_page, args=(zf.read(file),urlId,idx))

                if doc != -1:
                    json_array.append(doc)

            except Exception as e:
                print("parse error with reason : " + str(e) + " on page " + urlId + " filename " + file + "\n")
            continue


if __name__ == '__main__':
    readUpdate(sys.argv)
