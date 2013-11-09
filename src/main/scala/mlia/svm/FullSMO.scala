package mlia.svm

import scala.annotation.tailrec
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Uniform

object FullSMO {

  type Mat = DenseMatrix[Double]

  case class OptStruct(dataMat: Mat,
                       labelMat: Mat,
                       alphas: Mat,
                       b: Double = 0.0,
                       constant: Double,
                       tolerance: Double) {

    val eCache: Mat = DenseMatrix.zeros[Double](dataMat.rows, 2)

    val rows = dataMat.rows

    def label(i: Int): Double = labelMat(i, 0)

    def alpha(idx: Int) = alphas(idx, 0)

    def validEcacheArr: Array[Int] = eCache(::, 0).findAll(_ != 0).toArray

    def calcETA(i: Int, j: Int): Double = {
      ((((dataMat(i, ::) * dataMat(j, ::).t: Mat) :* 2.0: Mat) :-
        dataMat(i, ::) * dataMat(i, ::).t: Mat) :- dataMat(j, ::) *
        dataMat(j, ::).t: Mat)(0, 0)
    }

    def calcEk(k: Int): Double = {
      val fXk: Mat = ((alphas :* labelMat: Mat).t * (dataMat * dataMat(k, ::).t: Mat): Mat) :+ b
      fXk(0, 0) - label(k)
    }

    def nonBoundIndices: Seq[Int] = alphas.findAll(x => x > 0 && x < constant).map(_._1)

    def cache(i: Int, ei: Double) {
      eCache(i, ::) := DenseVector(1, ei)
    }

    def updateEk(k: Int) {
      cache(k, calcEk(k))
    }

    def newB(b: Double) = {
      copy(b = b)
    }

    case class JOpt(maxK: Int = dataMat.rows - 1, maxDeltaE: Double = 0.0, ej: Double = 0.0)

    def selectJ(i: Int, ei: Double): (Int, Double) = {
      cache(i, ei)

      if (validEcacheArr.size > 1) {
        // loop through valid Ecache values and find the one that maximizes delta E
        val opt = validEcacheArr.filter(_ != i).foldLeft(JOpt()) { (jOpt, k) =>
          val ek = calcEk(k)
          val deltaE = (ei - ek).abs
          if (deltaE > jOpt.maxDeltaE) JOpt(k, deltaE, ek) else jOpt
        }
        (opt.maxK, opt.ej)
      } else {
        val j = selectJrand(i, rows)
        val ej = calcEk(j)
        (j, ej)
      }
    }
  }

  def innerL(i: Int, oS: OptStruct) = {
    val ei = oS.calcEk(i)
    if ((oS.label(i) * ei < -oS.tolerance && oS.alpha(i) < oS.constant) || (oS.label(i) * ei > oS.tolerance && oS.alpha(i) > 0)) {
      val (j, ej) = oS.selectJ(i, ei)
      val (alphaIold, alphaJold) = (oS.alpha(i), oS.alpha(j))
      val (low, high) = calcLH(oS, i, j)
      if (low == high) {
        println(s"L == H[$low]")
        (0, oS)
      } else {
        // calculate optimal amount to change alpha[j]
        val eta = oS.calcETA(i, j)
        if (eta >= 0) {
          println(s"eta[$eta] >= 0")
          (0, oS)
        } else {
          oS.alphas(j, 0) -= oS.label(j) * (ei - ej) / eta
          oS.alphas(j, 0) = clipAlpha(oS.alpha(j), high, low)
          oS.updateEk(j)
          if ((oS.alpha(j) - alphaJold).abs < 0.00001) {
            println(s"j not moving enough[${(oS.alpha(j) - alphaJold).abs}]")
            (0, oS)
          } else {
            oS.alphas(i, 0) += (oS.label(j) * oS.label(i) * (alphaJold - oS.alpha(j)))
            oS.updateEk(i)

            val (b1, b2) = calcB(oS, i, alphaIold, ei, j, alphaJold, ej)
            val newB = {
              if (oS.alpha(i) > 0 && oS.alpha(i) < oS.constant) b1
              else if (oS.alpha(j) > 0 && oS.alpha(j) < oS.constant) b2
              else (b1 + b2) / 2.0
            }
            (1, oS.newB(newB))
          }
        }
      }
    } else (0, oS)
  }

  def smoP(dataMatIn: Array[Array[Double]], classLabels: Array[Double], c: Double, toler: Double, maxIter: Int): (Mat, Double) = {

    @tailrec
    def outerL(oS: OptStruct, iter: Int = 0, entireSet: Boolean = true, curAlphaPairsChanged: Int = 0): (Mat, Double) = {
      if (iter >= maxIter || (curAlphaPairsChanged == 0 && !entireSet)) {
        (oS.alphas, oS.b)
      } else {
        val (alphaPairsChanged, updatedOS) = if (entireSet) {
          // go over all values
          Range(0, oS.rows).foldLeft(0, oS) { case ((totalChanged, curOS), i) =>
            val (changed, newOS) = innerL(i, curOS)
            println(s"fullSet, iter: $iter i:$i, pairs changed $totalChanged")
            (totalChanged + changed, newOS)
          }
        } else {
          // go over non-bound (railed) alphas
          oS.nonBoundIndices.foldLeft(0, oS) { case ((totalChanged, curOS), i) =>
            val (changed, newOS) = innerL(i, curOS)
            println(s"non-bound, iter: $iter i:$i, pairs changed $totalChanged")
            (totalChanged + changed, newOS)
          }
        }
        // toggle entire set loop
        val updatedEntireSet = if (entireSet) false else if (alphaPairsChanged == 0) true else entireSet
        println(s"iteration number: $iter")
        outerL(updatedOS, iter + 1, updatedEntireSet, alphaPairsChanged)
      }
    }

    outerL(OptStruct(
      dataMat = DenseMatrix(dataMatIn: _*),
      labelMat = DenseMatrix(classLabels).t,
      alphas = DenseMatrix.zeros[Double](dataMatIn.size, 1),
      b = 0.0, constant = c, tolerance = toler))
  }

  private def calcB(os: OptStruct, i: Int, alphaIold: Double, ei: Double, j: Int, alphaJold: Double, ej: Double): (Double, Double) = {

    val b1 = os.b - ei - (os.label(i) * (os.alpha(i) - alphaIold)) *
      (os.dataMat(i, ::) * os.dataMat(i, ::).t: Mat)(0, 0) -
      (os.label(j) * (os.alpha(j) - alphaJold)) * (os.dataMat(i, ::) *
        os.dataMat(j, ::).t: Mat)(0, 0)

    val b2 = os.b - ej - (os.label(i) * (os.alpha(i) - alphaIold)) *
      (os.dataMat(i, ::) * os.dataMat(j, ::).t: Mat)(0, 0) -
      (os.label(j) * (os.alpha(j) - alphaJold)) * (os.dataMat(j, ::) *
        os.dataMat(j, ::).t: Mat)(0, 0)
    (b1, b2)
  }

  private def calcLH(os: OptStruct, i: Int, j: Int): (Double, Double) = {
    if (os.label(i) != os.label(j)) {
      val dev = os.alpha(j) - os.alpha(i)
      (scala.math.max(0.0, dev), scala.math.min(os.constant, os.constant + dev))
    } else {
      val total = os.alpha(j) + os.alpha(i)
      (scala.math.max(0.0, total - os.constant), scala.math.min(os.constant, total))
    }
  }

  def calcWs(alphas: Mat, dataArr: Seq[Array[Double]], classLabels: Array[Double]) = {
    val x = DenseMatrix(dataArr: _*)
    val labelMat = DenseMatrix(classLabels).t
    Range(0, x.rows).foldLeft(DenseMatrix.zeros[Double](x.cols, 1)) { (state, i) =>
      state :+ (alphas(i, ::) :* labelMat(i, ::): Mat) * x(i, ::)
    }
  }

  // the following code is the same as SimplifiedSMO

  def selectJrand(i: Int, m: Int): Int = {
    val rand = Uniform(0, m)
    @tailrec
    def step(i: Int, j: Int): Int = if (i != j) j else step(i, rand.sample().toInt)
    step(i, rand.sample().toInt)
  }

  def clipAlpha(aj: Double, h: Double, l: Double): Double = if (aj > h) h else if (aj < l) l else aj
}
