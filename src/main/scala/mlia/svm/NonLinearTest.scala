package mlia.svm

import FullSMOWithKernel._
import breeze.linalg.DenseMatrix

object NonLinearTest {

  type Mat = DenseMatrix[Double]

  def calcErrorRate(trainPath: String, evalPath: String, k1: Double = 1.3) = {

    val (dataArr, labelArr) = Prep.loadDataSet(trainPath)
    val kernel = Kernel("rbf", Array(k1))
    val weights = smoP(dataArr.toArray, labelArr.toArray, 200, 0.0001, 10000, kernel)

    val evaluator = Evaluator(DenseMatrix(dataArr: _*), DenseMatrix(labelArr.toArray).t, weights, kernel)
    println(s"there are ${evaluator.svAlphas.rows} Support Vectors")

    // calculate error rate of training data
    val trainErrorRate = evaluator.trainErrorRate
    println(f"the training error rate is: $trainErrorRate%.5f")

    // calculate error rate of test data with model made by training data
    val (evalDataArr, evalLabelArr) = Prep.loadDataSet(evalPath)
    val testErrorRate = evaluator.test(DenseMatrix(evalDataArr: _*), DenseMatrix(evalLabelArr.toArray).t)
    println(f"the test error rate is: $testErrorRate%.5f")
  }

  case class Evaluator(dataMat: Mat, labelMat: Mat, weights: (Mat, Double), kernel: Kernel) {

    val (alphas, b) = weights
    val svInd = alphas.findAll(_ > 0.0).map(_._1)
    val sVs = svInd.zipWithIndex.foldLeft(DenseMatrix.zeros[Double](svInd.size, dataMat.cols)) { case (state, (ind, seq)) =>
      state(seq, ::) := dataMat(ind, ::); state
    }

    // get matrix of only support vectors
    val svAlphas = filter(alphas, svInd).t
    val labelSV = filter(labelMat, svInd).t

    val trainErrorRate = test(dataMat, labelMat)

    def test(data: Mat, label: Mat): Double = errorCount(data, label).toDouble / data.rows

    private def errorCount(data: Mat, label: Mat): Int = {
      (0 until data.rows).foldLeft(0) { (state, i) =>
        val kernelEval = kernelTrans(sVs, data(i, ::), kernel)
        val predict = ((kernelEval.t * (labelSV :* svAlphas): Mat) :+ b: Mat)(0, 0)
        if (predict.signum != label(i, 0).signum) state + 1 else state
      }
    }

    private def filter(mat: DenseMatrix[Double], svInd: Seq[Int]): DenseMatrix[Double] =
      DenseMatrix(mat.iterator.filter { case (idx, _) => svInd.contains(idx._1) }.map(_._2).toArray)
  }

}

