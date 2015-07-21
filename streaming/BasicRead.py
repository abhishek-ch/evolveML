
from __future__ import print_function
__author__ = 'abc'
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
if __name__ == "__main__":

    sc = SparkContext("local[2]",appName="PythonStreamingHDFSWordCount")
    ssc = StreamingContext(sc, 1)
    lines = ssc.textFileStream('/Volumes/work/data/kaggle/test/')
    lines.pprint()
    counts = lines.flatMap(lambda line: line.split(" "))\
                  .map(lambda x: (x, 1))\
                  .reduceByKey(lambda a, b: a+b)
    counts.pprint()
    ssc.start()
    ssc.awaitTermination()