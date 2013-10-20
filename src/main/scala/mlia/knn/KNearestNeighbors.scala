package mlia.knn

import breeze.linalg._
import breeze.numerics._

object KNearestNeighbors {

  def classify0(inX: Vector[Double], dataSet: Matrix[Double], labels: Vector[String], k: Int): String = {

    require(k > 0)
    require(dataSet.rows == labels.size)
    require(inX.size == dataSet.cols)

    val sqDiff: DenseMatrix[Double] = tile(inX, dataSet.rows) - dataSet.toDenseMatrix :^ 2.0
    val distances = sqrt(sum(sqDiff, Axis._1))
    val sortedDistIndices = distances.activeIterator.toArray.sortBy(_._2).take(k)
    val classCount = sortedDistIndices.foldLeft(Map.empty[String, Int]) { (map, dist) =>
      val vote = labels(dist._1)
      map + (vote -> (map.getOrElse(vote, 0) + 1))
    }
    classCount.toArray.sortBy(_._2).reverse.headOption.map(_._1).getOrElse("Failure")
  }

  private def tile(in: Vector[Double], repeat: Int): DenseMatrix[Double] = {
    val mat = DenseMatrix.zeros[Double](repeat, in.size)
    for (i <- 1 to repeat) mat(i - 1, ::) := in
    mat
  }
}
