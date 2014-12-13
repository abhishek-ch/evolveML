__author__ = 'abhishekchoudhary'
import sys
# sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')

sys.path.append('/usr/local/lib/python2.7/site-packages/')
import hadoopy
from mrjob.job import MRJob

import os


def readHDFS(path):
     for value in hadoopy.readtb(path):
         print "aoooo".value


readHDFS("hdfs:///tmp/ACT6240.TXT")