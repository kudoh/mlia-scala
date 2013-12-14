package mlia.regression

import breeze.linalg._
import breeze.numerics._

object Regression {

  type Mat = DenseMatrix[Double]

  /**
   * Liner regression.
   */
  def standRegres(xArr: Array[Array[Double]], yArr: Array[Double]): Mat = {
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
  def lwlr(testPoint: Array[Double], xArr: Array[Array[Double]], yArr: Array[Double], k: Double = 1.0): Mat = {
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

  /**
   * Ridge regression.
   */
  def ridgeRegres(xMat: Mat, yMat: Mat, lam: Double = 0.2): Mat = {
    val xTx: Mat = xMat.t * xMat
    val denom = xTx + diag(DenseVector.ones[Double](xMat.cols)) * lam
    if (det(denom) == 0) {
      println("This matrix is singular, cannot do inverse")
      DenseMatrix.zeros[Double](0, 0)
    } else {
      inv(denom) * (xMat.t * yMat)
    }
  }

  def ridgeTest(xArr: Array[Array[Double]], yArr: Array[Double]): Mat = {
    val xMat = DenseMatrix(xArr: _*)
    val yMat = DenseMatrix(yArr).t
    val yMean = mean(yMat)
    // to eliminate X0 take mean off of Y
    val yDev: Mat = yMat - yMean

    val xReg = regularize(xMat)

    (0 until 30).foldLeft(DenseMatrix.zeros[Double](30, xMat.cols)) { (state, i) =>
      val ws = ridgeRegres(xReg, yDev, scala.math.exp(i - 10))
      state(i, ::) := ws.t
      state
    }
  }

  /**
   * utility: regularize by columns.
   */
  def regularize(xMat: Mat): Mat = {
    // calc mean then subtract it off
    val xMeans: DenseMatrix[Double] = mean(xMat, Axis._0)
    // calc variance of Xi then divide by it
    val xVar = variance(xMat, Axis._0)
    (0 until xMat.rows).foldLeft(DenseMatrix.zeros[Double](xMat.rows, xMat.cols)) { (state, i) =>
      state(i, ::) := (xMat(i, ::) - xMeans) / xVar
      state
    }
  }

  case class StageWiseState(ws: Mat, wsMax: Mat, lowestError: Double = Inf) {
    def initError(): StageWiseState = copy(lowestError = Inf)
  }

  object StageWiseState {
    def apply(ws: Mat): StageWiseState = StageWiseState(ws, ws)
  }

  /**
   * Forward stagewise regression.
   */
  def stageWise(xArr: Array[Array[Double]], yArr: Array[Double], eps: Double = 0.01, numIt: Int = 100): Mat = {
    val xMat = DenseMatrix(xArr: _*)
    val yMat = DenseMatrix(yArr).t
    val yMean = mean(yMat)
    val yDev: Mat = yMat - yMean
    val xReg = regularize(xMat)

    (0 until numIt).foldLeft(StageWiseState(DenseMatrix.zeros[Double](xMat.cols, 1))) { (outerState, i) =>
      println(outerState.ws.t)
      val curState = (0 until xMat.cols).foldLeft(outerState.initError()) { (innerState, j) =>
        Array(-1, 1).foldLeft(innerState) { (state, sign) =>
          val wsTest = state.ws.copy
          wsTest(j, 0) += eps * sign
          val yTest: Mat = xReg * wsTest
          val rssE = rssError(yDev.valuesIterator.toArray, yTest.valuesIterator.toArray)
          if (rssE < state.lowestError) state.copy(lowestError = rssE, wsMax = wsTest) else state
        }
      }
      curState.copy(ws = curState.wsMax)
    }
  }.ws
}
