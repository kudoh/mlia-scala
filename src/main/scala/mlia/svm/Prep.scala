package mlia.svm

import java.io.{File, BufferedReader, FileReader}

object Prep {

  def loadDataSet(fileName: String): (Seq[Array[Double]], Seq[Double]) = withIterator(fileName) { iter =>
    iter.map { line =>
      val lineArr = line.split('\t')
      (Array(lineArr(0).toDouble, lineArr(1).toDouble), lineArr(2).toDouble)
    }.toArray.unzip
  }

  private def withIterator[R](fileName: String)(f: Iterator[String] => R) = {
    val reader = new BufferedReader(new FileReader(new File(getClass.getResource(fileName).toURI)))
    try {f(Iterator.continually(reader.readLine()).takeWhile(_ != null))} finally {reader.close() }
  }
}
