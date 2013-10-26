package mlia.lr

import breeze.linalg._
import breeze.numerics._

object LogisticRegression {

  val alpha = 0.001
  val maxCycle = 500
  val sigmoid = breeze.generic.UFunc((x: Double) => 1.0 / (1 + exp(-1 * x)))

  def gradAscent(dataMatIn: List[Array[Double]], classLabels: Array[Int]) = {
    val dataMatrix = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseVector(classLabels.map(_.toDouble))
    Range(1, maxCycle).foldLeft(DenseMatrix.ones[Double](dataMatrix.cols, 1)) { (curWeight, cycle) =>
      val dot1: DenseVector[Double] = dot(dataMatrix, curWeight)
      val h: DenseVector[Double] = sigmoid(dot1)
      if (cycle == 100) println(dot1)
      val error: DenseVector[Double] = labelMat :- h
      val addition: DenseMatrix[Double] = dataMatrix.t :* alpha
      val dotError: DenseVector[Double] = dot(addition, error.toDenseMatrix)
      val weights = curWeight :+ dotError.toDenseMatrix
      if (cycle % 100 == 0) println(s"$cycle : ${weights.toDenseVector.toString()}")
      weights
    }
  }

  def dot(x: DenseMatrix[Double], y: DenseMatrix[Double]): DenseVector[Double] =
    DenseVector(Range(0, x.rows).map(i => (x(i, ::) :* y.t).sum).toArray)
}
