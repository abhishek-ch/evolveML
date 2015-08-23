/**
 * Created by abc on 23/08/2015.
 */

import org.apache.spark.SparkConf
import org.apache.spark.streaming.{Seconds, StreamingContext}

object DirStreamCount {
  def main(args: Array[String]) {


    //StreamingExamples.setStreamingLogLevels()
    val sparkConf = new SparkConf().setAppName("HdfsWordCount")
    // Create the context
    val ssc = new StreamingContext(sparkConf, Seconds(2))

    // Create the FileInputDStream on the directory and use the
    // stream to count words in new files created
    val lines = ssc.textFileStream("hdfs://127.0.0.1:50075/spark/test")
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1)).reduceByKey(_ + _)
    print("OK LETS SEE")
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }

}
