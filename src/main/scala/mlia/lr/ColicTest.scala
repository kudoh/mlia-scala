package mlia.lr

import java.io.{FileReader, File, BufferedReader}
import breeze.linalg.DenseVector


object ColicTest {

  def calcErrorRate(trainingFileName: String, testingFileName: String)
                   (trainer: (List[Array[Double]], Array[Int]) => DenseVector[Double]) = withIterator(trainingFileName, testingFileName) { (iterTrain, iterTest) =>

    def split(line: String) = line.split('\t')
    def predictors(arr: Array[String]) = (for (i <- 0 to 21) yield arr(i).toDouble).toArray
    def target(arr: Array[String]) = arr(21).toDouble.toInt

    // train model and finally get weights.
    val (trainingSet, trainingLabels) = iterTrain.foldLeft(List.empty[Array[Double]] -> List.empty[Int]) { case ((curTrain, curLabels), line) =>
      val arr = split(line)
      (curTrain :+ predictors(arr)) -> (curLabels :+ target(arr))
    }
    val trainWeights = trainer(trainingSet, trainingLabels.toArray)

    // test model using testing file.
    val (totalCount, errorCount) = iterTest.foldLeft(0, 0) { case ((curCnt, curErrCnt), line) =>
      val arr = split(line)
      val classified = LogisticRegression.classifyVector(DenseVector(predictors(arr)), trainWeights)
      if (classified != target(arr)) (curCnt + 1, curErrCnt + 1) else (curCnt + 1, curErrCnt)
    }

    // calculate error rate
    val errorRate = errorCount.toDouble / totalCount
    println(f"the error rate of this test is: $errorRate%.6f")
    errorRate
  }

  def calcErrorRateMean(trainingFileName: String, testingFileName: String, numTests: Int = 10)
                       (trainer: (List[Array[Double]], Array[Int]) => DenseVector[Double]) = {
    val errorSum = (0 until numTests).foldLeft(0.0) { (curErrorSum, k) =>
      curErrorSum + calcErrorRate(trainingFileName, testingFileName)(trainer)
    }
    println(f"after $numTests%d iterations the average error rate is: ${errorSum / numTests.toDouble}%.6f")
  }

  def withIterator[R](trainingFileName: String, testingFileName: String)(f: (Iterator[String], Iterator[String]) => R) = {

    def reader(name: String) = new BufferedReader(new FileReader(new File(getClass.getResource(name).toURI)))
    def iterator(reader: BufferedReader) = Iterator.continually(reader.readLine()).takeWhile(_ != null)

    val reader1 = reader(trainingFileName)
    val reader2 = reader(testingFileName)
    try {
      f(iterator(reader1), iterator(reader2))
    } finally {reader1.close(); reader2.close() }
  }
}
