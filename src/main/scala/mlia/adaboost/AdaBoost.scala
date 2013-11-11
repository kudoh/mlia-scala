package mlia.adaboost

import breeze.linalg._
import breeze.numerics._
import scala.annotation.tailrec

object AdaBoost {

  type Mat = DenseMatrix[Double]

  def stumpClassify(dataMat: Mat, dimen: Int, threshVal: Double, threshIneq: String): Mat = {
    val retArray = DenseMatrix.ones[Double](dataMat.rows, 1)
    threshIneq match {
      case "lt" =>
        val bi = dataMat(::, dimen) :<= threshVal
        (0 until dataMat.rows).foreach(i => if (bi(i)) retArray(i, 0) = -1.0)
      case _ =>
        val bi = dataMat(::, dimen) :> threshVal
        (0 until dataMat.rows).foreach(i => if (bi(i)) retArray(i, 0) = -1.0)
    }
    retArray
  }

  case class Stump(dim: Int = -1, threshold: Double = 0.0,
                   ineq: String = "", bestClassEst: Mat,
                   minError: Double = Double.PositiveInfinity,
                   alpha: Double = 0.0) {
    override def toString = s"dim:$dim, threshold:$threshold, ineqal:$ineq, minErr:$minError, bestClassEst:[${bestClassEst.t}]"
  }

  def buildStump(dataArr: Array[Array[Double]], classLabels: Array[Double], D: Mat): Stump = {
    val dataMat = DenseMatrix(dataArr: _*)
    val labelMat = DenseMatrix(classLabels)
    val numSteps = 10.0
    val initialRet = Stump(bestClassEst = DenseMatrix.zeros[Double](dataMat.rows, 1))

    (0 until dataMat.cols).foldLeft(initialRet) { (outer, i) =>
      val rangeMin = dataMat(::, i).min
      val rangeMax = dataMat(::, i).max
      val stepSize = (rangeMax - rangeMin) / numSteps

      // loop over all range in current dimension
      (-1 to numSteps.toInt).foldLeft(outer) { (inner, j) =>
        @tailrec
        def calc(remaining: List[String], cur: Stump): Stump = remaining match {
          case List() => cur
          case List(inequal, _*) =>
            val threshVal = rangeMin + j.toDouble * stepSize
            // call stump classify with i, j, lessThan
            val predictedVals = stumpClassify(dataMat, i, threshVal, inequal)

            val errArr = DenseMatrix.ones[Double](dataMat.rows, 1)
            val error = predictedVals(::, 0) :== labelMat(::, 0)
            (0 until errArr.rows).foreach(i => if (error(i)) errArr(i, 0) = 0)
            // calc total error multiplied by D
            val weightedError = (D.t * errArr: Mat)(0, 0)
            // println(f"split: dim $i, thresh $threshVal%.2f, thresh ineqal: $inequal, the weighted error is $weightedError%.3f")

            calc(remaining.tail, if (weightedError < inner.minError) Stump(i, threshVal, inequal, predictedVals, weightedError) else cur)
        }

        // go over less than and greater than
        calc(List("lt", "gt"), inner)
      }
    }
  }

  case class Result(D: DenseMatrix[Double], aggClassEst: DenseMatrix[Double], weakClassArr: Array[Stump] = Array.empty) {
    override def toString = s"D: ${D.t}, aggClassEst: ${aggClassEst.t}, weakClassArr: ${weakClassArr.mkString(",")}"
  }

  def adaBoostTrainDS(dataArr: Array[Array[Double]], classLabels: Array[Double], numIt: Int = 40) = {

    val labelMat = DenseMatrix(classLabels).t

    @tailrec
    def loop(state: Result, i: Int, curErrorRate: Double): Result = {
      if (curErrorRate == 0.0) {
        println("Error Rate is 0.")
        state
      } else if (i == numIt) {
        println("Iteration finish.")
        state
      } else {
        val stump = buildStump(dataArr, classLabels, state.D)
        println(s"D: ${state.D.t}")

        val alpha = 0.5 * log((1.0 - stump.minError) / scala.math.max(stump.minError, 1e-16))
        val newStump = stump.copy(alpha = alpha)

        println(s"classEst: ${stump.bestClassEst.t}")

        val expon = (labelMat * (-1 * alpha): Mat) :* newStump.bestClassEst
        val tempD = state.D :* exp(expon)
        val newD = tempD / tempD.sum
        state.aggClassEst :+= (newStump.bestClassEst * alpha)
        println(s"aggClassEst: ${state.aggClassEst.t}")
        val aggErrors = (0 until state.aggClassEst.rows).foldLeft(DenseMatrix.zeros[Double](labelMat.rows, 1)) { (errors, i) =>
          errors(i, 0) = if (state.aggClassEst(i, 0).signum == labelMat(i, 0)) 0.0 else 1.0; errors
        }
        val errorRate: Double = aggErrors.sum / dataArr.size
        println(s"total error: $errorRate")

        loop(Result(newD, state.aggClassEst, state.weakClassArr :+ newStump), i + 1, errorRate)
      }
    }

    val initialResult =
      Result(DenseMatrix.ones[Double](dataArr.size, 1) / dataArr.size.toDouble,
        DenseMatrix.zeros[Double](dataArr.size, 1))

    loop(initialResult, 0, 100.0)
  }

  def adaClassify(datToClass: Array[Array[Double]], classifierArr: Array[Stump]) = {

    val dataMatrix = DenseMatrix(datToClass: _*)
    val initialEst = DenseMatrix.zeros[Double](dataMatrix.rows, 1)
    val weights = classifierArr.zipWithIndex.foldLeft(initialEst) { case (aggClassEst, (classifier, i)) =>
      val classEst = stumpClassify(dataMatrix, classifier.dim, classifier.threshold, classifier.ineq)
      aggClassEst :+= classEst * classifier.alpha
      println(aggClassEst)
      aggClassEst
    }
    signum(weights)
  }
}
