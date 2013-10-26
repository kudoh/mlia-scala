package mlia.lr

import java.io.{FileReader, BufferedReader, File}


object Prep {
  def loadDataSet(fileName: String): (List[Array[Double]], List[Int]) = withReader(fileName) { reader =>
    Iterator.continually(reader.readLine()).takeWhile(_ != null).map { line =>
      val lineArr = line.split('\t')
      (Array(1.0, lineArr(0).toDouble, lineArr(1).toDouble), lineArr(2).toInt)
    }.toList.unzip
  }

  private def withReader[R](fileName: String)(f: BufferedReader => R) = {
    val reader = new BufferedReader(new FileReader(new File(getClass.getResource(fileName).toURI)))
    try {f(reader)} finally {reader.close() }
  }
}
