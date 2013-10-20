package mlia.trees

import breeze.numerics._

object Trees {

  case class Row(data: Array[Int], label: String)

  case class InformationGain(featureIdx: Int, infoGain: Double)

  def calcShannonEnt(dataSet: Array[Row]) = {

    val labelCounts = dataSet.foldLeft(Map.empty[String, Int]) { (map, row) =>
      map + (row.label -> (map.getOrElse(row.label, 0) + 1))
    }
    val numEntries = dataSet.size
    labelCounts.foldLeft(0.0) { (state, count) =>
      val prob = labelCounts(count._1).toDouble / numEntries
      state - prob * (log(prob) / log(2))
    }
  }

  def splitDataSet(dataSet: Array[Row], axis: Int, value: Int) = dataSet.filter(_.data(axis) == value)

  def chooseBestFeatureToSplit(dataSet: Array[Row]) = {

    val numEntries = dataSet.size
    val numFeatures = dataSet.head.data.size
    val baseEntropy = calcShannonEnt(dataSet)

    Range(0, numFeatures).foldLeft(InformationGain(-1, 0.0)) { (curBest, cur) =>
      val uniqueVals = dataSet.map(_.data(cur)).distinct
      val newEntropy = uniqueVals.foldLeft(0.0) { (ent, value) =>
        val subDataSet = splitDataSet(dataSet, cur, value)
        val prob = subDataSet.size / numEntries.toDouble
        ent + prob * calcShannonEnt(subDataSet)
      }
      val infoGain = baseEntropy - newEntropy
      if (infoGain > curBest.infoGain) InformationGain(cur, infoGain) else curBest
    }
  }

  def majorityCnt(classList: Array[String]): String =
    classList.foldLeft(Map.empty[String, Int]) { (state, x) =>
      state + (x -> (state.getOrElse(x, 0) + 1))
    }.toArray.sortBy(_._2).reverse.head._1

  private def remove(num: Int, list: Array[String]) = list diff Array(num)

  case class Node(key: String, value: Any, children: Array[Node] = Array.empty) {
    override def toString =
      if (children.isEmpty) s" -> $value[Leaf]" else s"{$key : $value ${children.map(_.toString).mkString(",")}}"
  }

  def createTree(dataSet: Array[Row], labels: Array[String],
                 cur: Array[Node] = Array.empty, value: Int = -1): Array[Node] = {

    val classList = dataSet.map(_.label)
    if (classList.distinct.size == 1) cur :+ Node(value.toString, classList(0)) // all label is equal
    else if (dataSet.head.data.isEmpty) cur :+ Node(value.toString, majorityCnt(classList)) // no more feature
    else {
      val bestFeat = chooseBestFeatureToSplit(dataSet).featureIdx
      val subLabels = remove(bestFeat, labels)
      val uniqueFeatValues = dataSet.map(_.data(bestFeat)).distinct
      uniqueFeatValues.map { value =>
        cur :+ Node(labels(bestFeat), value.toString,
          createTree(splitDataSet(dataSet, bestFeat, value), subLabels, cur, value))
      }.flatten
    }
  }
}
