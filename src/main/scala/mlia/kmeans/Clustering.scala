package mlia.kmeans

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Uniform

object Clustering {

  type Vec = DenseVector[Double]
  type Mat = DenseMatrix[Double]

  def distEuclid(vecA: Vec, vecB: Vec) = sqrt(sum((vecA - vecB) :^ 2.0: Vec))

  def randCent(dataSet: Mat, k: Int) = {
    (0 until dataSet.cols).foldLeft(DenseMatrix.zeros[Double](k, dataSet.cols)) { (centroids, j) =>
      val data: Vec = dataSet(::, j)
      val minJ = data.min
      val rangeJ = data.max - minJ
      centroids(::, j) := DenseVector(new Uniform(0, 1).sample(k).map(_ * rangeJ + minJ): _*)
      centroids
    }
  }
}
