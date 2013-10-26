package mlia.lr

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Uniform

object LogisticRegression {

  val sigmoid = breeze.generic.UFunc((x: Double) => 1.0 / (1 + exp(-1 * x)))

  def gradAscent(dataMatIn: List[Array[Double]], classLabels: Array[Int]): DenseVector[Double] = {

    val alpha = 0.001
    val maxCycle = 500
    val dataMatrix = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseVector(classLabels.map(_.toDouble))

    Range(0, maxCycle).foldLeft(DenseMatrix.ones[Double](dataMatrix.cols, 1)) { (curWeight, cycle) =>
      val h: DenseVector[Double] = sigmoid(dot(dataMatrix, curWeight))
      val error: DenseVector[Double] = labelMat :- h
      val weights = curWeight :+ dot(dataMatrix.t :* alpha, error.toDenseMatrix).toDenseMatrix
      if (cycle % 100 == 0) println(s"$cycle : ${weights.toDenseVector.toString()}")
      weights
    }.toDenseVector
  }

  def stocGradAscent0(dataMatIn: List[Array[Double]], classLabels: Array[Int]): DenseVector[Double] = {
    val alpha = 0.01
    dataMatIn.zipWithIndex.foldLeft(DenseVector.ones[Double](dataMatIn.head.size)) { (curWeight, row) =>
      val (value, i) = row
      val vec = DenseVector(value)
      val h = sigmoid(dot(vec.toDenseMatrix, curWeight.toDenseMatrix).sum)
      val error = classLabels(i) - h
      curWeight :+ (vec :* (alpha * error): DenseVector[Double])
    }
  }

  def stocGradAscent1(dataMatIn: List[Array[Double]], classLabels: Array[Int], numIter: Int = 150): DenseVector[Double] = {

    Range(0, numIter).foldLeft(DenseVector.ones[Double](dataMatIn.head.size)) { (outerState, i) =>
      Range(0, dataMatIn.size).foldLeft((outerState, Range(0, dataMatIn.size).toArray)) {
        case ((curWeights, indices), j) =>
          val alpha = (4 / (1.0 + i + j)) + 0.01
          val randIndex = Uniform(0, indices.size).sample().toInt
          val vec = DenseVector(dataMatIn(randIndex))

          val h = sigmoid((vec :* curWeights: DenseVector[Double]).sum)
          val error = classLabels(randIndex) - h

          val newWeights = (curWeights :+ (vec :* (alpha * error): DenseVector[Double]), indices.tail)
          if (i % 10 == 0 && j % 100 == 0) println(s"$i, $j : $newWeights")
          newWeights
      }._1
    }
  }

  private def dot(x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] =
    DenseVector(Range(0, x.rows).map(i => (x(i, ::) :* y.t).sum).toArray)
}
