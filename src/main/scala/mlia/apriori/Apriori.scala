package mlia.apriori

import scala.collection.mutable

object Apriori {

  def createC1(dataSet: Array[Array[Int]]): Set[Int] = dataSet.flatten.toSet

  def scanD(D: Array[Set[Int]], Ck: Set[Int], minSupport: Double): (Array[Int], Map[Int, Double]) = {

    val ssCnt = D.foldLeft(Map.empty[Int, Int]) { (outer, tran) =>
      Ck.foldLeft(outer)((inner, item) => if (Set(item).subsetOf(tran)) inner + (item -> (inner.getOrElse(item, 0) + 1)) else inner)
    }

    ssCnt.foldLeft(Array.empty[Int], Map.empty[Int, Double]) { case ((retValues, supportData), (can, cnt)) =>
      val support = cnt.toDouble / D.length
      (if (support >= minSupport) can +: retValues else retValues,
        supportData + (can -> support))
    }
  }
}
