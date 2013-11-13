package mlia.regression

import java.io.{FileReader, BufferedReader, File}

object Prep {

  def loadDataSet(fileName: String): (Array[Array[Double]], Array[Double]) = withIterator(fileName) { (ite, numFeat) =>
    val dataAndLabels = ite.toArray.map { line =>
      val lineArr = line.split('\t').map(_.toDouble)
      val feat = (0 until numFeat - 1).map(i => lineArr(i))
      val label = lineArr(numFeat - 1)
      (feat, label)
    }.unzip

    dataAndLabels._1.map(_.toArray).toArray -> dataAndLabels._2.toArray
  }

  private def withIterator[R](fileName: String)(f: (Iterator[String], Int) => R) = {
    val file = new File(getClass.getResource(fileName).toURI)
    val headerReader = new BufferedReader(new FileReader(file))
    val numFeat = try {headerReader.readLine().split('\t').size} finally {headerReader.close() }
    val bodyReader = new BufferedReader(new FileReader(file))
    try {f(Iterator.continually(bodyReader.readLine()).takeWhile(_ != null), numFeat)} finally {headerReader.close() }
  }
}
