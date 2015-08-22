/**
 * Created by abc on 22/08/2015.
 */
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf

object ReadStream {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Reading Zip & Contents")
    val sc = new SparkContext(sparkConf)

    val readZip = sc.wholeTextFiles("/Volumes/work/data/kaggle/dato/test/5.zip").cache()

    val contents = readZip.flatMap({
      case (name , contents) => unzip(contents)
    })

    val names = readZip.flatMap({
      case (name , contents) => unzip(name)
    })





    contents.take(5)
    sc.stop()
  }


  def unzip(contents : String) : String = {
    println("=> "+contents)
    return contents
  }
}
