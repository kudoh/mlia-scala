package mlia.cart

import breeze.linalg._
import scala.annotation.tailrec

object Cart {

  type Mat = DenseMatrix[Double]

  case class TreeNode(spInd: Int = -1, spVal: Double, left: Option[TreeNode] = None, right: Option[TreeNode] = None) {

    override def toString: String = if (!isLeaf) s"[feature: $spInd, threshold: $spVal, left: ${left.getOrElse("-")}, , right: ${right.getOrElse("-")}]" else s"$spVal"

    val isLeaf = spInd == -1
  }

  case class Row(data: Array[Double])

  case class DataSet(rows: Array[Row]) extends Seq[Row] {

    val length: Int = rows.size

    val iterator: Iterator[Row] = rows.iterator

    lazy val allSameLabel: Boolean = rows.map(_.data.last).distinct.size == 1

    def foldPredictors[R](z: R)(op: (R, Double, Int) => R): R = rows.map(_.data.slice(0, colSize - 1)).foldLeft(z) { (outer, elem) =>
      elem.zipWithIndex.foldLeft(outer) { case (inner, (value, colIdx)) => op(inner, value, colIdx) }
    }

    val colSize: Int = if (rows.isEmpty) -1 else rows.head.data.size

    def apply(row: Int): Row = this.row(row)

    def apply(row: Int, col: Int): Double = rows(row).data(col)

    def cell(row: Int, col: Int): Double = apply(row, col)

    def row(row: Int): Row = rows(row)

    def binSplitDataSet(feature: Int, value: Double): (DataSet, DataSet) = {

      @tailrec
      def split(i: Int, left: DataSet, right: DataSet): (DataSet, DataSet) = {
        if (i == length) (left, right)
        else {
          val newLeft = if (cell(i, feature) > value) left :+ row(i) else left
          val newRight = if (cell(i, feature) <= value) right :+ row(i) else right
          split(i + 1, DataSet(newLeft.toArray), DataSet(newRight.toArray))
        }
      }
      split(0, DataSet.empty, DataSet.empty)
    }

    def createTree(ops: Array[Double] = Array.empty)(implicit model: Model): TreeNode = {
      val (feat, value) = chooseBestSplit(ops)
      feat.map { idx =>
        val (lSet, rSet) = binSplitDataSet(idx, value)
        TreeNode(idx, value, Some(lSet.createTree(ops)), Some(rSet.createTree(ops)))
      } getOrElse TreeNode(-1, value)
    }

    case class BestSplitCtx(bestS: Double = Double.PositiveInfinity, bestIndex: Int = 0, bestValue: Double = 0.0)

    def chooseBestSplit(ops: Array[Double] = Array(1, 4))(implicit model: Model): (Option[Int], Double) = {

      val Array(tolS, tolN, _*) = ops
      // if all the target variables are the same value: quit and return value
      if (allSameLabel) (None, model.getLeaf(this))
      else {
        // the choice of the best feature is driven by Reduction in RSS error from mean
        val S = model.calcError(this)
        val finalCtx = foldPredictors(BestSplitCtx()) { (curCtx, splitVal, featIndex) =>
          val (left, right) = binSplitDataSet(featIndex, splitVal)
          if (left.length < tolN || right.length < tolN) curCtx
          else {
            val newS = model.calcError(left) + model.calcError(right)
            if (newS < curCtx.bestS) BestSplitCtx(bestS = newS, bestIndex = featIndex, bestValue = splitVal) else curCtx
          }
        }

        // if the decrease (S-bestS) is less than a threshold don't do the split
        if ((S - finalCtx.bestS) < tolS) {
          (None, model.getLeaf(this))
        } else {
          val (left2, right2) = this.binSplitDataSet(finalCtx.bestIndex, finalCtx.bestValue)
          if (left2.length < tolN || right2.length < tolN) {
            (None, model.getLeaf(this))
          } else {
            (Some(finalCtx.bestIndex), finalCtx.bestValue)
          }
        }
      }
    }
  }

  object DataSet {

    def empty: DataSet = new DataSet(Array.empty)

    def apply(arr: Array[Array[Double]]): DataSet = DataSet(arr.map(elem => Row(elem)))

    implicit val regModel = Regression
  }

  trait Model {

    def getLeaf(dataSet: DataSet): Double

    def calcError(dataSet: DataSet): Double
  }

  object Regression extends Model {

    def getLeaf(dataSet: DataSet): Double = mean(dataSet.rows.map(_.data.last))

    def calcError(dataSet: DataSet): Double = variance(dataSet.rows.map(_.data.last)) * dataSet.length
  }

}
