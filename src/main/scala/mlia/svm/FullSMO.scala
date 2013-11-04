package mlia.svm

import breeze.linalg._
import breeze.numerics._

object FullSMO {

  case class OptStruct(dataMat: DenseMatrix[Double],
                       labelMat: DenseMatrix[Double],
                       alphas: DenseMatrix[Double],
                       b: Double = 0.0,
                       constant: Double,
                       tolerance: Double,
                       eCache: DenseMatrix[Double]) {

    val rows = dataMat.rows

    def add(i: Int, ei: Double): OptStruct = {
      eCache(i, ::) := DenseVector(1, ei)
      copy(eCache = eCache)
    }

    def updateEk(k: Int): OptStruct = {
       val ek = calcEk(this, k)
       add(k, ek)
    }
  }

  def calcEk(oS: OptStruct, k: Int) = {
    val fXk: DenseMatrix[Double] = ((oS.alphas :* oS.labelMat: DenseMatrix[Double]).t * (oS.dataMat * oS.dataMat(k, ::).t: DenseMatrix[Double]): DenseMatrix[Double]) :+ oS.b
    fXk(0, 0) - oS.labelMat(k, 0)
  }

  case class JOpt(maxK: Int = -1, maxDeltaE: Double = 0.0, ej: Double = 0.0)

  def selectJ(i: Int, curOS: OptStruct, ei: Double) = {

    val oS = curOS.add(i, ei)
    val validEcacheList: Array[Int] = oS.eCache(::, 0).findAll(_ != 0).toArray

    if (validEcacheList.size > 1) {
      val opt = validEcacheList.filter(_ != i).foldLeft(JOpt()) { (jOpt, k) =>
        val ek = calcEk(oS, k)
        val deltaE = abs(ei - ek)
        if (deltaE > jOpt.maxDeltaE) JOpt(k, deltaE, ek) else jOpt
      }
      (opt.maxK, opt.ej, oS)
    } else {
      val j = SimplifiedSMO.selectJrand(i, oS.rows)
      println(j)
      val ej = calcEk(oS, j)
      (j, ej, oS)
    }
  }
}
