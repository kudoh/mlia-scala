package mlia.cart

import breeze.linalg._
import scala.annotation.tailrec

object Cart {

  type Mat = DenseMatrix[Double]

  case class TreeNode(spInd: Int = -1, spVal: Double, left: Option[TreeNode] = None, right: Option[TreeNode] = None) {

    override def toString: String = if (!isLeaf) s"[feature: $spInd, threshold: $spVal, left: ${left.getOrElse("-")}, right: ${right.getOrElse("-")}]" else s"$spVal"

    val isLeaf = spInd == -1

    val isTree = !isLeaf

    def mean: Double = {
      ((right map (r => if (r.isTree) r.mean else spVal)).getOrElse(spVal) +
        (left map (l => if (l.isTree) l.mean else spVal)).getOrElse(spVal)) / 2.0
    }

    def leftValue = left.filter(_.isLeaf).map(_.spVal).getOrElse(throw new IllegalStateException())

    def rightValue = right.filter(_.isLeaf).map(_.spVal).getOrElse(throw new IllegalStateException())

    def prune(testData: DataSet): TreeNode = {
      if (testData.length == 0) TreeNode(-1, mean)
      else {
        val curTree = if (right.exists(_.isTree) || left.exists(_.isTree)) {
          val (lSet, rSet) = testData.binSplitDataSet(spInd, spVal)
          val maybeLeft = left.filter(_.isTree).map(_.prune(lSet)).orElse(left)
          val maybeRight = right.filter(_.isTree).map(_.prune(rSet)).orElse(right)
          copy(left = maybeLeft, right = maybeRight)
        } else this

        if (curTree.right.exists(_.isLeaf) && curTree.left.exists(_.isLeaf)) {
          val (lSet, rSet) = testData.binSplitDataSet(curTree.spInd, curTree.spVal)
          val errorNoMerge = square(lSet.labelMat - curTree.leftValue).sum + square(rSet.labelMat - curTree.rightValue).sum
          val treeMean = (curTree.leftValue + curTree.rightValue) / 2.0
          val errorMerge = square(testData.labelMat - treeMean).sum
          if (errorMerge < errorNoMerge) {
            println("merging")
            TreeNode(-1, treeMean)
          } else curTree
        } else curTree
      }
    }

    def square(m: DenseMatrix[Double]): DenseMatrix[Double] = m.map(x => scala.math.pow(x, 2))
  }

  case class Row(data: Array[Double])

  case class DataSet(rows: Array[Row]) extends Seq[Row] {

    val length: Int = rows.size

    val iterator: Iterator[Row] = rows.iterator

    lazy val allSameLabel: Boolean = rows.map(_.data.last).distinct.size == 1

    lazy val labelMat: DenseMatrix[Double] = DenseMatrix(rows.map(_.data.last))

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

    def createTree(ops: Array[Double])(implicit model: Model): TreeNode = {
      val (feat, value) = chooseBestSplit(ops)
      feat.map { idx =>
        val (lSet, rSet) = binSplitDataSet(idx, value)
        TreeNode(idx, value, Some(lSet.createTree(ops)), Some(rSet.createTree(ops)))
      } getOrElse TreeNode(-1, value)
    }

    case class BestSplitCtx(bestS: Double = Double.PositiveInfinity, bestIndex: Int = 0, bestValue: Double = 0.0)

    def chooseBestSplit(ops: Array[Double])(implicit model: Model): (Option[Int], Double) = {

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
        if ((S - finalCtx.bestS) < tolS) (None, model.getLeaf(this))
        else {
          val (left2, right2) = this.binSplitDataSet(finalCtx.bestIndex, finalCtx.bestValue)
          if (left2.length < tolN || right2.length < tolN) (None, model.getLeaf(this))
          else {
            println(finalCtx.bestIndex + ":" + finalCtx.bestValue)
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

    def calcError(dataSet: DataSet): Double = {
      val labels = dataSet.rows.map(_.data.last)
      val avg = mean(labels)
      labels.map(x => scala.math.pow(x - avg, 2)).sum
    }
  }

}
