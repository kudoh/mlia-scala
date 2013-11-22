package mlia.cart

import breeze.linalg._
import scala.annotation.tailrec
import scala.collection.mutable

object Cart {

  type Mat = DenseMatrix[Double]

  abstract class Threshold

  abstract class TreeNode[T](val spInd: Int, val spVal: T, val left: Option[TreeNode[T]] = None, val right: Option[TreeNode[T]] = None) {

    override def toString: String = if (!isLeaf) s"[feature: $spInd, threshold: $spVal, left: ${left.getOrElse("-")}, right: ${right.getOrElse("-")}]" else s"$spVal"

    val isLeaf = spInd == -1

    val isTree = !isLeaf

    val doubleValue: Double

    def mean: Double

    def branch(left: Option[TreeNode[T]], right: Option[TreeNode[T]]): TreeNode[T]

    def leftValue: T = left.filter(_.isLeaf).map(_.spVal).getOrElse(throw new IllegalStateException())

    def rightValue: T = right.filter(_.isLeaf).map(_.spVal).getOrElse(throw new IllegalStateException())

    def prune(testData: DataSet): TreeNode[T]

    def square(m: DenseMatrix[Double]): DenseMatrix[Double] = m.map(x => scala.math.pow(x, 2))

    def wrapDouble(d: Double): T
  }

  case class RegTree(override val spInd: Int,
                     override val spVal: Double,
                     override val left: Option[TreeNode[Double]] = None,
                     override val right: Option[TreeNode[Double]] = None) extends TreeNode[Double](spInd, spVal, left, right) {

    def wrapDouble(d: Double): Double = d

    val doubleValue: Double = spVal

    def branch(l: Option[TreeNode[Double]], r: Option[TreeNode[Double]]): TreeNode[Double] = this.copy(left = l, right = r)

    def mean: Double =
      (right.map(r => if (r.isTree) r.mean else spVal).getOrElse(spVal) +
        left.map(l => if (l.isTree) l.mean else spVal).getOrElse(spVal)) / 2.0

    def prune(testData: DataSet): TreeNode[Double] = {
      if (testData.length == 0) RegTree(-1, mean)
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
            RegTree(-1, treeMean)
          } else curTree
        } else curTree
      }
    }

  }

  case class ModelTree(override val spInd: Int,
                       override val spVal: Mat,
                       override val left: Option[TreeNode[Mat]] = None,
                       override val right: Option[TreeNode[Mat]] = None) extends TreeNode[Mat](spInd, spVal, left, right) {

    val doubleValue: Double = spVal(0, 0)

    def wrapDouble(d: Double): Mat = DenseMatrix(d)

    def branch(l: Option[TreeNode[Cart.Mat]], r: Option[TreeNode[Cart.Mat]]): TreeNode[Cart.Mat] = this.copy(left = l, right = r)

    def mean: Double = {
      import breeze.linalg.{mean => matMean}
      (right.map(r => if (r.isTree) r.mean else matMean(spVal)).getOrElse(matMean(spVal)) +
        left.map(l => if (l.isTree) l.mean else matMean(spVal)).getOrElse(matMean(spVal))) / 2.0
    }

    def prune(testData: DataSet): TreeNode[Mat] = {
      if (testData.length == 0) ModelTree(-1, wrapDouble(mean))
      else {
        val curTree = if (right.exists(_.isTree) || left.exists(_.isTree)) {
          val (lSet, rSet) = testData.binSplitDataSet(spInd, doubleValue)
          val maybeLeft = left.filter(_.isTree).map(_.prune(lSet)).orElse(left)
          val maybeRight = right.filter(_.isTree).map(_.prune(rSet)).orElse(right)
          copy(left = maybeLeft, right = maybeRight)
        } else this

        if (curTree.right.exists(_.isLeaf) && curTree.left.exists(_.isLeaf)) {
          val (lSet, rSet) = testData.binSplitDataSet(curTree.spInd, curTree.doubleValue)
          val errorNoMerge = square(lSet.labelMat - curTree.leftValue).sum + square(rSet.labelMat - curTree.rightValue).sum
          val treeMean = (curTree.leftValue + curTree.rightValue) / 2.0
          val errorMerge = square(testData.labelMat - treeMean).sum
          if (errorMerge < errorNoMerge) {
            println("merging")
            ModelTree(-1, treeMean)
          } else curTree
        } else curTree
      }
    }

  }

  case class Row(data: Array[Double])

  case class DataSet(rows: Array[Row]) extends Seq[Row] {

    val length: Int = rows.size

    val iterator: Iterator[Row] = rows.iterator

    lazy val allSameLabel: Boolean = rows.map(_.data.last).distinct.size == 1

    lazy val labelMat: DenseMatrix[Double] = DenseMatrix(rows.map(_.data.last))

    lazy val dataMat: DenseMatrix[Double] = DenseMatrix(rows.map(x => x.data.slice(1, rows.length - 1)): _*)

    def foldPredictors[R](z: R)(op: (R, Double, Int) => R): R = rows.map(_.data.slice(0, colSize - 1)).foldLeft(z) { (outer, elem) =>
      elem.zipWithIndex.foldLeft(outer) { case (inner, (value, colIdx)) => op(inner, value, colIdx) }
    }

    val colSize: Int = if (rows.isEmpty) -1 else rows.head.data.size

    def apply(row: Int): Row = this.row(row)

    def apply(row: Int, col: Int): Double = rows(row).data(col)

    def cell(row: Int, col: Int): Double = apply(row, col)

    def row(row: Int): Row = rows(row)

    def createTree[T](ops: Array[Double])(implicit op: TreeOps[T] with TreeBuilder[T]): TreeNode[T] = {
      val (feat, value) = chooseBestSplit(ops)
      println(s"feat:$feat, value:$value")
      feat.map { idx =>
        val (lSet, rSet) = binSplitDataSet(idx, op.toDouble(value))
        op.branch(idx, value, Some(lSet.createTree(ops)), Some(rSet.createTree(ops)))
      } getOrElse op.leaf(value)
    }

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

    def chooseBestSplit[T](ops: Array[Double])(implicit model: TreeOps[T]): (Option[Int], T) = {

      val Array(tolS, tolN, _*) = ops
      // if all the target variables are the same value: quit and return value
      if (allSameLabel) (None, model.getLeaf(this))
      else {
        // the choice of the best feature is driven by Reduction in RSS error from mean
        val S = model.calcError(this)
        val finalCtx = foldPredictors(BestSplitCtx(Double.PositiveInfinity, -1, 0.0)) { (curCtx, splitVal, featIndex) =>
          val (left, right) = binSplitDataSet(featIndex, splitVal)
          if (left.length < tolN || right.length < tolN) curCtx
          else {
            val newS = model.calcError(left) + model.calcError(right)
            if (newS < curCtx.bestS) BestSplitCtx(bestS = newS, bestIndex = featIndex, bestValue = splitVal) else curCtx
          }
        }
        println(finalCtx)
        // if the decrease (S-bestS) is less than a threshold don't do the split
        if ((S - finalCtx.bestS) < tolS) (None, model.getLeaf(this))
        else {
          val (left2, right2) = binSplitDataSet(finalCtx.bestIndex, finalCtx.bestValue)
          if (left2.length < tolN || right2.length < tolN) (None, model.getLeaf(this))
          else {
            println(finalCtx.bestIndex + ":" + finalCtx.bestValue)
            (Some(finalCtx.bestIndex), model.doubleToValue(finalCtx.bestValue))
          }
        }
      }
    }
  }

  object DataSet {

    def empty: DataSet = new DataSet(Array.empty)

    def apply(arr: Array[Array[Double]]): DataSet = DataSet(arr.map(elem => Row(elem)))
  }

  case class BestSplitCtx[Leaf](bestS: Leaf, bestIndex: Int = 0, bestValue: Leaf)

  object Regression extends RegOps with RegTreeBuilder

  object Model extends ModelOps with ModelTreeBuilder

  trait TreeOps[A] {

    def getLeaf(dataSet: DataSet): A

    def calcError(dataSet: DataSet): Double

    def doubleToValue(threshold: Double): A

    def toDouble(value: A): Double
  }

  trait RegOps extends TreeOps[Double] {

    def getLeaf(dataSet: DataSet): Double = mean(dataSet.rows.map(_.data.last))

    def calcError(dataSet: DataSet): Double = {
      val labels = dataSet.rows.map(_.data.last)
      val avg = mean(labels)
      labels.map(x => scala.math.pow(x - avg, 2)).sum
    }

    def doubleToValue(threshold: Double): Double = threshold

    def toDouble(value: Double): Double = value
  }

  trait ModelOps extends TreeOps[Mat] {

    def getLeaf(dataSet: DataSet): Mat = linearSolve(dataSet)._1

    def calcError(dataSet: DataSet): Double = {
      println(s"ds:${dataSet.length},${dataSet.colSize}")
      val (ws, x, y) = linearSolve(dataSet)
      val yHat = x * ws
      (y - yHat: Mat).map(x => scala.math.pow(x, 2)).sum
    }

    def linearSolve(dataSet: DataSet): (DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {
      val X = DenseMatrix.ones[Double](dataSet.length, dataSet.colSize)
      val Y = dataSet.labelMat.t
      X(::, 1 until X.cols) := dataSet.dataMat
      val xTx = X.t * X
      if (det(xTx) == 0)
        throw new IllegalStateException("This matrix is singular, cannot do inverse, try increasing the second value of ops")
      val ws = inv(xTx) * (X.t * Y)
      (ws, X, Y)
    }

    def doubleToValue(threshold: Double): Cart.Mat = DenseMatrix(threshold)

    def toDouble(value: Mat) = value(0, 0)
  }

  trait TreeBuilder[T] {

    def branch(feature: Int, splitValue: T, left: Option[TreeNode[T]], right: Option[TreeNode[T]]): TreeNode[T]

    def leaf(splitVal: T): TreeNode[T]
  }

  trait RegTreeBuilder extends TreeBuilder[Double] {

    def leaf(splitVal: Double): TreeNode[Double] = this.branch(-1, splitVal, None, None)

    def branch(feature: Int, splitValue: Double, left: Option[TreeNode[Double]], right: Option[TreeNode[Double]]): TreeNode[Double] = RegTree(feature, splitValue, left, right)
  }

  trait ModelTreeBuilder extends TreeBuilder[Mat] {

    def leaf(splitVal: Mat): TreeNode[Mat] = this.branch(-1, splitVal, None, None)

    def branch(feature: Int, splitValue: Cart.Mat, left: Option[TreeNode[Cart.Mat]], right: Option[TreeNode[Cart.Mat]]): TreeNode[Cart.Mat] = ModelTree(feature, splitValue, left, right)
  }

}
