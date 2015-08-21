/**
 * Created by abc on 20/08/2015.
 */


import org.apache.spark._

object Read{
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Reading Zip & Contents").setMaster("local[2]")
    val sc = new SparkContext(sparkConf)

    val readZip = sc.wholeTextFile("/Volumes/work/data/kaggle/dato/test/5.zip")

    val contents = readZip.flatMap({
      case (name , contents) => unzip(contents)
    })

    sc.stop()
  }


  def unzip(contents : String) : String = {
    println("=> "+contents)
    return ""
  }
}

