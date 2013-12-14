package mlia.kmeans

import scala.annotation.tailrec
import breeze.linalg._
import breeze.numerics._
import breeze.stats.distributions.{Rand, RandBasis, Uniform}

object Clustering {

  type Vec = DenseVector[Double]
  type Mat = DenseMatrix[Double]

  implicit def distEuclid(vecA: Vec, vecB: Vec): Double = sqrt(sum((vecA - vecB) :^ 2.0: Vec))

  implicit def randCent(dataSet: Mat, k: Int)(implicit rand: RandBasis = Rand): Mat = {
    (0 until dataSet.cols).foldLeft(DenseMatrix.zeros[Double](k, dataSet.cols)) { (centroids, j) =>
      val data: Vec = dataSet(::, j)
      val minJ = data.min
      val rangeJ = data.max - minJ
      centroids(::, j) := DenseVector(new Uniform(0, 1).sample(k).map(_ * rangeJ + minJ): _*)
      centroids
    }
  }

  object KMeans {

    def apply(dataSet: Mat, k: Int)(implicit distMeans: (Vec, Vec) => Double, createCent: (Mat, Int) => Mat) = {

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
        import scala.collection.mutable.ArrayBuffer
        val changed = clusterAssment(i).clusterIndex != updateAssment.clusterIndex
        val buf = ArrayBuffer(clusterAssment: _*)
        buf.update(i, updateAssment)
        copy(clusterAssment = buf.toArray, clusterChanged = changed)
      }

      def getIndices(cent: Int) = clusterAssment.zipWithIndex.filter { case (ass, i) => ass.clusterIndex == cent}.map(_._2)

      override def toString = s"centroid:\n $centroids\ndataPoints:${clusterAssment.mkString(", ")}"
    }

    case class RowState(minDist: Double = Inf, closestIndex: Int = -1)

  }

  object BisectingKMeans {

    def apply(dataSet: Mat, k: Int)(implicit distMeans: (Vec, Vec) => Double, createCent: (Mat, Int) => Mat) = {

      val centroid0 = mean(dataSet, Axis._0)
      // calculate initial Error and update initial state
      val initialState = (0 until dataSet.rows).foldLeft(State(Array(centroid0), Array.fill(dataSet.rows)(Assessment.zero))) { (state, j) =>
        val error = distMeans(centroid0.toDenseVector, dataSet(j, ::).toDenseVector)
        state.update(j, Assessment(j, error))
        state
      }

      @tailrec
      def iterate(state: State, curK: Int): State = {
        if (curK == k) {
          println("number of k is satisfied.")
          state
        } else {
          val bestSplit = (0 until state.centroids.size).foldLeft(SplitState()) { (inner, centIdx) =>
          // get the data points currently in cluster centIdx
            val ptsInClust = DenseMatrix(state.getIndices(centIdx).map(i => dataSet(i, ::).valuesIterator.toArray): _*)
            val clustered = KMeans(ptsInClust, 2)
            // compare the SSE to the currrent minimum
            val sseSplit = clustered.clusterAssment.map(_.error).sum
            val sseNotSplit = state.clusterAssment.filter(_.clusterIndex != centIdx).map(_.error).sum
            println(f"sseSplit: $sseSplit%.5f, sseNotSplit: $sseNotSplit%.5f")
            if (sseSplit + sseNotSplit < inner.lowestSSE) {
              SplitState(centIdx, clustered.centroids, clustered.clusterAssment, sseSplit + sseNotSplit)
            } else inner
          }
          // reassign cluster index to original index. because split index is 0/1
          val splitAss = bestSplit.reassignIndex(state.centroids.size)

          println(s"the bestCentToSplit is: ${bestSplit.bestCentToSplit}")
          println(s"the len of bestClustAss is: ${bestSplit.bestClustAss.length}")

          // update one centroid into old one
          import scala.collection.mutable.ArrayBuffer
          val buf = ArrayBuffer[Mat](state.centroids: _*)
          buf.update(bestSplit.bestCentToSplit, bestSplit.bestNewCents(0, ::))
          // append another centroid
          buf += bestSplit.bestNewCents(1, ::)
          // replace new cluster assessment(error) into current centroid
          val zippedAss = splitAss zip state.getIndices(bestSplit.bestCentToSplit)
          val newAss = state.clusterAssment.zipWithIndex.map { case (curAss, i) =>
            if (curAss.clusterIndex == bestSplit.bestCentToSplit) zippedAss.find(_._2 == i).map(_._1).head else curAss
          }

          iterate(State(buf.toArray, newAss), curK + 1)
        }
      }

      iterate(initialState, 1)
    }

    case class State(centroids: Array[Mat], clusterAssment: Array[Assessment]) {

      // mutate function
      def update(i: Int, updateAssment: Assessment): State = {
        import scala.collection.mutable.ArrayBuffer
        val buf = ArrayBuffer(clusterAssment: _*)
        buf.update(i, updateAssment)
        copy(clusterAssment = buf.toArray)
      }

      def getIndices(cent: Int): Array[Int] =
        clusterAssment.zipWithIndex.filter { case (ass, i) => ass.clusterIndex == cent}.map(_._2)

      override def toString = s"centroid:\n $centMat\ndataPoints:${clusterAssment.mkString(", ")}"

      def centMat = DenseMatrix(centroids.map(_.valuesIterator.toArray): _*)
    }

    case class SplitState(bestCentToSplit: Int = -1, bestNewCents: Mat = DenseMatrix.zeros[Double](0, 0),
                          bestClustAss: Array[Assessment] = Array.empty, lowestSSE: Double = Inf) {

      def reassignIndex(newIdx: Int) = bestClustAss.map { ass =>
        if (ass.clusterIndex == 1) new Assessment(newIdx, ass.error) else new Assessment(bestCentToSplit, ass.error)
      }
    }

  }

  final class Assessment(val clusterIndex: Int, val error: Double) {

    override def toString = f"[clusterIndex: $clusterIndex, error: $error]"

    override def equals(other: scala.Any): Boolean = other match {
      case that: Assessment => this.clusterIndex == that.clusterIndex && this.error == that.error
      case _ => false
    }

    override def hashCode(): Int = 31 * clusterIndex.hashCode() + error.hashCode()
  }

  object Assessment {

    def apply(index: Int, dist: Double) = new Assessment(index, scala.math.pow(dist, 2))

    def zero = new Assessment(0, 0)
  }

}
