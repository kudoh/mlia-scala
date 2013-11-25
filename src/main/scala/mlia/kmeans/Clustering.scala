package mlia.kmeans

import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.Uniform
import scala.collection.mutable.ArrayBuffer
import scala.annotation.tailrec

object Clustering {

  type Vec = DenseVector[Double]
  type Mat = DenseMatrix[Double]

  implicit def distEuclid(vecA: Vec, vecB: Vec): Double = sqrt(sum((vecA - vecB) :^ 2.0: Vec))

  implicit def randCent(dataSet: Mat, k: Int): Mat = {
    (0 until dataSet.cols).foldLeft(DenseMatrix.zeros[Double](k, dataSet.cols)) { (centroids, j) =>
      val data: Vec = dataSet(::, j)
      val minJ = data.min
      val rangeJ = data.max - minJ
      centroids(::, j) := DenseVector(new Uniform(0, 1).sample(k).map(_ * rangeJ + minJ): _*)
      centroids
    }
  }

  def kMeans(dataSet: Mat, k: Int)(implicit distMeans: (Vec, Vec) => Double, createCent: (Mat, Int) => Mat) = {

    @tailrec
    def iterate(state: State): State = {
      if (!state.clusterChanged) {
        println("centroid does not moved anymore.")
        state
      } else {
        println("centroid moved. calculate distance between each point and centroid...")
        // for each data point assign it to the closest centroid
        val outerResult = (0 until dataSet.rows).foldLeft(state) { (outer, i) =>
          val innerResult = (0 until k).foldLeft(RowState()) { (inner, clusterIdx) =>
            val distJI = distMeans(outer.centroids(clusterIdx, ::).toDenseVector, dataSet(i, ::).toDenseVector)
            if (distJI < inner.minDist) RowState(distJI, clusterIdx) else inner
          }
          outer.update(i, Assessment(innerResult.closestIndex, innerResult.minDist))
        }
        val centroid = (0 until k).foldLeft(DenseMatrix.zeros[Double](k, dataSet.cols)) { (curCentroid, cent) =>
          val ptsInClust = DenseMatrix(outerResult.getIndices(cent).map(i => dataSet(i, ::).valuesIterator.toArray): _*)
          curCentroid(cent, ::) := mean(ptsInClust, Axis._0)
          curCentroid
        }
        iterate(State(centroid, outerResult.clusterAssment, outerResult.clusterChanged))
      }
    }

    iterate(State(createCent(dataSet, k), Array.fill(dataSet.rows)(Assessment.zero), clusterChanged = true))
  }

  case class State(centroids: Mat, clusterAssment: Array[Assessment], clusterChanged: Boolean) {

    def update(i: Int, updateAssment: Assessment) = {
      val changed = clusterAssment(i).clusterIndex != updateAssment.clusterIndex
      val buf = ArrayBuffer(clusterAssment: _*)
      buf.update(i, updateAssment)
      copy(clusterAssment = buf.toArray, clusterChanged = changed)
    }

    def getIndices(cent: Int) = clusterAssment.zipWithIndex.filter { case (ass, i) => ass.clusterIndex == cent }.map(_._2)

    override def toString = s"centroid:\n $centroids\ndataPoints:${clusterAssment.mkString(", ")}"
  }

  case class RowState(minDist: Double = Inf, closestIndex: Int = -1)

  class Assessment(val clusterIndex: Int, val error: Double) {
    override def toString = f"[clusterIndex: $clusterIndex, error: $error%.4f]"
  }

  object Assessment {

    def apply(index: Int, dist: Double) = new Assessment(index, scala.math.pow(dist, 2))

    def zero = new Assessment(0, 0)
  }

}
