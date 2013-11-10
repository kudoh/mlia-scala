package mlia.lr

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Uniform

object LogisticRegression {

  type Mat = DenseMatrix[Double]
  type Vec = DenseVector[Double]

  val sigmoid = breeze.generic.UFunc((x: Double) => 1.0 / (1 + exp(-1 * x)))

  def gradAscent(dataMatIn: List[Array[Double]], classLabels: Array[Int]): Vec = {

    val alpha = 0.001
    val maxCycle = 500
    val dataMatrix = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseMatrix(classLabels.map(_.toDouble)).t

    (0 until maxCycle).foldLeft(DenseMatrix.ones[Double](dataMatrix.cols, 1)) { (curWeight, cycle) =>
      val h = sigmoid(dataMatrix * curWeight)
      val error: Mat = labelMat :- h
      curWeight :+ (dataMatrix.t :* alpha: Mat) * error
    }.toDenseVector
  }

  def stocGradAscent0(dataMatIn: List[Array[Double]], classLabels: Array[Int]): Vec = {
    val alpha = 0.01
    dataMatIn.zipWithIndex.foldLeft(DenseVector.ones[Double](dataMatIn.head.size)) { case (curWeight, (row, i)) =>
      val vec = DenseVector(row)
      val h = sigmoid((vec.toDenseMatrix * curWeight.toDenseMatrix.t: Mat).sum)
      val error = classLabels(i) - h
      curWeight :+ (vec :* (alpha * error): Vec)
    }
  }

  def stocGradAscent1Iter500(dataMatIn: List[Array[Double]], classLabels: Array[Int]) = stocGradAscent1(dataMatIn, classLabels, 500)

  def stocGradAscent1(dataMatIn: List[Array[Double]], classLabels: Array[Int], numIter: Int = 150): Vec = {

    (0 until numIter).foldLeft(DenseVector.ones[Double](dataMatIn.head.size)) { (outerState, i) =>
      (0 until dataMatIn.size).foldLeft((outerState, (0 until dataMatIn.size).toArray)) { case ((curWeights, indices), j) =>
        val alpha = (4 / (1.0 + i + j)) + 0.01
        val randIndex = Uniform(0, indices.size).sample().toInt
        val vec = DenseVector(dataMatIn(randIndex))

        val h = sigmoid((vec :* curWeights: Vec).sum)
        val error = classLabels(randIndex) - h

        val newWeights = (curWeights :+ (vec :* (alpha * error): Vec), indices.tail)
        newWeights
      }._1
    }
  }

  def classifyVector(inX: Vector[Double], weights: Vector[Double]) = {
    val prob = sigmoid((inX :* weights).sum)
    if (prob > 0.5) 1 else 0
  }
}
