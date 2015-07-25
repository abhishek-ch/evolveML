__author__ = 'abc'
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from math import sin, cos, sqrt, atan2, radians
import math
import multiprocessing as mp

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees).
    Source: http://gis.stackexchange.com/a/56589/15183
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km

def calculateDistance(coordinateA , coordinateB):
    lon1, lat1, lon2, lat2 = coordinateA[0],coordinateA[1],coordinateB[0],coordinateB[1]
    #print "lon1",lon1," lat1 ",lat1," lon2 ",lon2," lat2 ",lat2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    km = 6367 * c
    return km

def update(*coordinates):
    #output.put(sum(calculateDistance(coordinates[i],coordinates[i+1]) for i in range(len(coordinates)-1)))
    return sum(calculateDistance(coordinates[i],coordinates[i+1]) for i in range(len(coordinates)-1))
    #print output
    return output
    '''
    #print(coordinates)
    total = 0
    #print(range(len(coordinates)-1))
    if len(coordinates) > 1:
        for i in range(len(coordinates)-1):
            print coordinates[i]  ," - ",coordinates[i+1]
        #total= sum(calculateDistance(coordinates[i],coordinates[i+1]) for i in range(len(coordinates)-1))
    #print(total)
    return total
    '''

if __name__ == '__main__':
    # Define an output queue
    output = mp.Queue()

    # reading training data
    zf = zipfile.ZipFile('train.csv.zip')
    df = pd.read_csv(zf.open('train.csv'),chunksize = 1000,
                 iterator = True,converters={'POLYLINE': lambda x: json.loads(x)})
    #print df.head()
    columnValue = []
    pool = mp.Pool(processes=4)
    for data in df:

        results = [pool.apply(update,args=(coords)) for coords in data['POLYLINE']]
        '''
        for p in processes:
            p.start()

        # Exit the completed processes
        for p in processes:
            p.join()
        '''
        # Get process results from the output queue
        #results = [output.get() for p in processes]
        #print(results)

    print " Ll ",len(df)," vv ",len(results)

            # Run processes



    #df['POLYLINE'] = df['POLYLINE'].apply(lambda x: mp.Process(target=update,args=(x)))

    #df['POLYLINE'] = df['POLYLINE'].apply(lambda coordinates: sum(calculateDistance(coordinates[i],coordinates[i+1]) for i in range(len(coordinates)-1)) )
    #tot = sum(distance(l[i],l[i+1]) for i in range(len(l)-1))