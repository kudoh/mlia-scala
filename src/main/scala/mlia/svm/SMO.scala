package mlia.svm

import scala.annotation.tailrec
import breeze.stats.distributions.Uniform
import breeze.linalg._


object SMO {

  def selectJrand(i: Int, m: Int) = {
    @tailrec
    def step(i: Int, j: Int): Int = if (i != j) j else step(i, Uniform(0, m).sample().toInt)
    step(i, Uniform(0, m).sample().toInt)
  }

  def clipAlpha(aj: Double, h: Double, l: Double): Double = if (aj > h) h else if (aj < l) l else aj

  def smoSimple(dataMatIn: Array[Array[Double]], classLabels: Array[Double], c: Double, toler: Double, maxIter: Int) = {

    val dataMat = DenseMatrix(dataMatIn: _*)
    val labelMat = DenseMatrix(classLabels).t

    @tailrec
    def outerLoop(curNum: Int = 0,
                  alphas: DenseMatrix[Double] = DenseMatrix.zeros(dataMat.rows, 1),
                  b: Double = 0): (DenseMatrix[Double], Double) = {
      if (curNum == maxIter) (alphas, b)
      else {
        Range(0, dataMat.rows).foldLeft(alphas, b) { case ((curAl, curB), i) =>

          // this is our prediction of the class
          val fXi: DenseMatrix[Double] = predict(alphas, labelMat, dataMat, dataMat(i, ::).t, b)
          // error = fXi - real class
          val eiMat = fXi :- labelMat
          val ei = eiMat(0, 0)
          // println(s"fxi: $fXi, ei: $ei")

          if ((labelMat(i, 0) * ei < -toler && alphas(i, 0) < c) ||
            (labelMat(i, 0) * ei > toler && alphas(i, 0) > 0)) {
            // enter optimization
            val j = selectJrand(i, dataMat.rows)
            val fXj: DenseMatrix[Double] = predict(alphas, labelMat, dataMat, dataMat(j, ::).t, b)
            val ejMat = fXj :- labelMat
            val ej = ejMat(0, 0)
            val alphaIold = alphas(i, 0)
            val alphaJold = alphas(j, 0)

            // guarantee alphas stay between 0 and c
            val (low, high) = if (labelMat(i, 0) != labelMat(j, 0)) {
              val dev: Double = alphas(j, 0) - alphas(i, 0)
              (scala.math.max(0.0, dev), scala.math.min(c, c + dev))
            } else {
              val total: Double = alphas(j, 0) + alphas(i, 0)
              (scala.math.max(0.0, total - c), scala.math.min(c, total))
            }

            if (low == high) {
              println(s"L == H[$low]")
              (curAl, curB)
            } else {
              // calculate optimal amount to change alpha[j]
              val eta1: DenseMatrix[Double] = (dataMat(i, ::) * dataMat(j, ::).t: DenseMatrix[Double]) :* 2.0
              val eta2: DenseMatrix[Double] = eta1 :- dataMat(i, ::) * dataMat(i, ::).t
              val eta3: DenseMatrix[Double] = eta2 :- dataMat(j, ::) * dataMat(j, ::).t
              val eta = eta3(0, 0)
              if (eta >= 0) {
                println(s"eta[$eta] >= 0")
                (curAl, curB)
              } else {
                alphas(j, ::) :-= labelMat(j, 0) * (ei - ej) / eta
                alphas(j, ::) := clipAlpha(alphas(j, 0), high, low)
              }
            }
          }

          (curAl, curB)
        }
        outerLoop(curNum + 1, alphas, b)
      }
    }
    outerLoop(0)
  }


  def predict(alphas: DenseMatrix[Double], labelMat: DenseMatrix[Double], dataMat: DenseMatrix[Double],
              transposedRow: DenseMatrix[Double], b: Double): DenseMatrix[Double] = {
    ((alphas :* labelMat).t * (dataMat * transposedRow): DenseMatrix[Double]) :+ b
  }

  def multiply(x1: DenseMatrix[Double], x2: DenseMatrix[Double]) = x1.mapPairs {
    case ((row, _), value) => value * x2(row, 1)
  }
}
