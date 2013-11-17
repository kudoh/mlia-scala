package mlia.cart

import java.io.{FileReader, BufferedReader, File}

object Prep {

  def loadDataSet(fileName: String): Array[Array[Double]] = withIterator(fileName) { ite =>
    ite.map(line => line.stripMargin.split('\t').map(_.toDouble)).toArray
  }

  private def withIterator[R](fileName: String)(f: Iterator[String] => R): R = {
    val reader = new BufferedReader(new FileReader(new File(getClass.getResource(fileName).toURI)))
    try {f(Iterator.continually(reader.readLine()).takeWhile(_ != null))} finally {reader.close() }
  }
}
