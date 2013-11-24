package mlia.kmeans

import java.io.{File, FileReader, BufferedReader}

object Prep {

  def loadDataSet(fileName: String): Array[Array[Double]] = withIterator(fileName) { iter =>
    iter.map { line =>
      line.split('\t').map(_.toDouble)
    }.toArray
  }

  private def withIterator[R](fileName: String)(f: Iterator[String] => R) = {
    val reader = new BufferedReader(new FileReader(new File(getClass.getResource(fileName).toURI)))
    try {f(Iterator.continually(reader.readLine()).takeWhile(_ != null))} finally {reader.close() }
  }
}
