package mlia.adaboost

import breeze.linalg._
import scala.annotation.tailrec

object DecisionStump {

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

  case class BestStump(dim: Int = -1, threshold: Double = 0.0,
                       ineq: String = "", bestClasEst: Mat, minError: Double = Double.PositiveInfinity) {
    override def toString: String = s"dim:$dim, threshold:$threshold, ineqal:$ineq, minErr:$minError, \nbestClassEst:\n$bestClasEst"
  }

  def buildStump(dataArr: Array[Array[Double]], classLabels: Array[Double], D: Mat): BestStump = {
    val dataMat = DenseMatrix(dataArr: _*)
    val labelMat = DenseMatrix(classLabels)
    val numSteps = 10.0
    val initialRet = BestStump(bestClasEst = DenseMatrix.zeros[Double](dataMat.rows, 1))

    (0 until dataMat.cols).foldLeft(initialRet) { (outer, i) =>
      val rangeMin = dataMat(::, i).min
      val rangeMax = dataMat(::, i).max
      val stepSize = (rangeMax - rangeMin) / numSteps

      // loop over all range in current dimension
      (-1 to numSteps.toInt).foldLeft(outer) { (inner, j) =>
        @tailrec
        def calc(remaining: List[String], cur: BestStump): BestStump = remaining match {
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
            println(f"split: dim $i, thresh $threshVal%.2f, thresh ineqal: $inequal, the weighted error is $weightedError%.3f")

            calc(remaining.tail, if (weightedError < inner.minError) BestStump(i, threshVal, inequal, predictedVals, weightedError) else cur)
        }

        // go over less than and greater than
        calc(List("lt", "gt"), inner)
      }
    }
  }
}
