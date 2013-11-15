package mlia.regression

import breeze.linalg._
import breeze.numerics._

object Regression {

  type Mat = DenseMatrix[Double]

  /**
   * Liner regression.
   */
  def standRegres(xArr: Array[Array[Double]], yArr: Array[Double]): DenseMatrix[Double] = {
    val xMat = DenseMatrix(xArr: _*)
    val yMat = DenseMatrix(yArr).t
    val xTx = xMat.t * xMat
    if (det(xTx) == 0.0) {
      println("This matrix is singular, cannot do inverse")
      DenseMatrix.zeros[Double](0, 0)
    } else inv(xTx) * (xMat.t * yMat)
  }

  /**
   * Locally weighted liner regression.
   */
  def lwlr(testPoint: Array[Double], xArr: Array[Array[Double]], yArr: Array[Double], k: Double = 1.0): DenseMatrix[Double] = {
    val xMat = DenseMatrix(xArr: _*)
    val yMat = DenseMatrix(yArr).t
    val weights = (0 until xMat.rows).foldLeft(diag(DenseVector.ones[Double](xMat.rows))) { (curWeights, j) =>
      val diffMat = DenseMatrix(testPoint) - xMat(j, ::)
      curWeights(j, j) = exp((diffMat * diffMat.t: Mat)(0, 0) / (-2.0 * scala.math.pow(k, 2)))
      curWeights
    }
    val xTx = xMat.t * (weights * xMat)
    if (det(xTx) == 0.0) {
      println("This matrix is singular, cannot do inverse")
      DenseMatrix.zeros[Double](0, 0)
    } else {
      val ws = inv(xTx) * (xMat.t * (weights * yMat))
      DenseMatrix(testPoint) * ws
    }
  }

  def lwlrTest(testArr: Array[Array[Double]], xArr: Array[Array[Double]], yArr: Array[Double], k: Double = 1.0): DenseVector[Double] = {
    testArr.zipWithIndex.foldLeft(DenseVector.zeros[Double](testArr.size)) { case (yHat, (test, i)) =>
      yHat(i) = lwlr(test, xArr, yArr, k)(0, 0); yHat
    }
  }

  def rssError(yArr: Array[Double], yHatArr: Array[Double]): Double =
    yArr.zip(yHatArr).foldLeft(0.0) { case (state, (y, yHat)) => state + scala.math.pow(y - yHat, 2) }
}
