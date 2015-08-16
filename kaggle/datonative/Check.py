__author__ = 'abc'
import os, sys, logging, string, glob

fIn = glob.glob( '/Volumes/work/data/kaggle/dato' + '/*/*raw*')
json_array, last_bucket = [], str(0)
for idx, filename in enumerate(fIn):
    #print 'ID %s name %s' %(idx,filename)

    filenameDetails = filename.split("/")
    urlId = filenameDetails[-1].split('_')[0]
    bucket = filenameDetails[-2]

    print filenameDetails
    print urlId
    print bucket

    if bucket != last_bucket or filename==fIn[-1]:
        last_bucket = bucket