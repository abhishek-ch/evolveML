"""
Step 2

Read the file from disk, calculate the number of kilometers for each trip (POLYLINE), and write the results to disk.
The results set should have the POLYLINE column omitted, and a new TRIP_LENGTH column added. You may
choose any storage format for the output file(s), but it should be optimized for fast reading in step 3.

It creates a directory named output , where it saves all the output files

Step 3

Read the file(s), determine the data type of each column, and write the results to a log file. This step should be
performed in the fastest way possible.

it creates a log file named columnType.log and save each file , each column data type
"""
__author__ = 'abc aka ABHISHEK CHOUDHARY'

import zipfile
import pandas as pd
import json
import math
import multiprocessing as mp
import os
import glob
import logging
import time


def calculateDistance(coordinateA, coordinateB):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    :param coordinateA: Each pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]
    :param coordinateB:
    :return:
    """
    lon1, lat1, lon2, lat2 = coordinateA[0], coordinateA[1], coordinateB[0], coordinateB[1]
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km


def extractCoordinate(coordinates):
    """
    it extract each POLYLINE column data as list and then find the sum of total distance travelled by a trip.
    It find the distance between each trip of 15 sec block
    :param coordinates: list of coordinates  Each pair of coordinates is also identified by the same brackets as [LONGITUDE, LATITUDE]
    :return: total distance travelled
    """
    value = 0.00
    if len(coordinates) > 1:
        value = round(sum(calculateDistance(coordinates[i], coordinates[i + 1]) for i in range(len(coordinates) - 1)),
                      2)
    return value


def readFileType():
    """
    it stores the data type of each column of every output file
    :return:
    """
    logging.basicConfig(filename='columnType.log', level=logging.DEBUG)
    for file in glob.glob("output/*.csv"):
        df = pd.read_csv(file, index_col=None, header=0)
        logging.debug(file)
        logging.info(df.dtypes)
        # print df.dtypes


if __name__ == '__main__':
    start_time = time.time()

    # reading training data
    # directly reading the file from zip format
    zf = zipfile.ZipFile('train.csv.zip')
    CHUNKSIZE = 1000
    pool = mp.Pool(processes=8)
    #count maintains the chunk index that as well retained to save the file name
    count = 0
    #hold the name of the columns, removing POLYLINE
    allcolumns = []
    # create output directory in the current location with name output
    try:
        os.makedirs("output")
    except OSError:
        if not os.path.isdir("output"):
            raise

    """
    iterate through each file with a limited chunksize , currently set as 1000
    as tested, since 1000 , is not crossing the memory limit beyond 1gb considering
    optimized memory utilization of the ide or tool using

    while iterating , made sure to extract the POLYLINE column as list
    which further being passed to calculate the total distance travelled in a trip
    """
    for chunk in pd.read_csv(zf.open('train.csv'), chunksize=CHUNKSIZE,
                             converters={'POLYLINE': lambda x: json.loads(x)}):
        results = [pool.apply(extractCoordinate, args=(coords)) for coords in zip(chunk['POLYLINE'])]
        chunk['TRIP_LENGTH'] = results
        seq = ("output/data_", ".csv")
        filename = str(count).join(seq)
        #extract column names from the first chunk and retain the same
        if count == 0:
            allcolumns = list(chunk.columns.values)
            allcolumns.remove('POLYLINE')

            #save the content to csv file
        chunk.to_csv(filename, sep=',', encoding='utf-8', columns=allcolumns, index=False)
        count += 1
    #read the file type and log the same
    readFileType()
    print("--- %s seconds ---" % (time.time() - start_time))


"""
Describe shortly at the end of the code how you would have improved the solution, if you had more time.

I'd first like to optimise the file writing procedure.
Believing the 8-core , I would have tried to split the entire reading the csv and submitting the result in paralllize
way using core. Basic fundamental of memory allocation and parallel processing

I might have split all the tasks into very small 8 subtasks, where 6 tasks were responsible for reading the data
and 2 tasks were responsible for writing to file. Logically it must have been work more faster as then I would have
been utilizing the each core properly and it was truly meant to be parallelism.
"""