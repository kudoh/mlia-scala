package mlia.apriori

import scala.annotation.tailrec

object Apriori {

  def createC1(dataSet: Array[Array[Int]]): Array[ItemSet] = dataSet.flatten.distinct.map(ItemSet.apply)

  def scanD(D: Array[ItemSet], Ck: Array[ItemSet], minSupport: Double): (Array[ItemSet], Supports) = {
    D.foldLeft(Map.empty[ItemSet, Int]) { (outer, tran) =>
      Ck.foldLeft(outer) { (inner, item) => // count up occurrence
        if (item.subsetOf(tran)) inner + (item -> (inner.getOrElse(item, 0) + 1)) else inner
      }
    }.foldLeft(Array.empty[ItemSet], Supports.empty) { case ((retValues, supportData), (can, cnt)) =>
      val support = cnt.toDouble / D.length // calculate support
      (if (support >= minSupport) can +: retValues else retValues,
        supportData + (can -> support))
    }
  }

  /**
   * Creates Ck from Lk(k >= 2).
   */
  def aprioriGen(Lk: Array[ItemSet], k: Int): Array[ItemSet] = {
    (for {(x1, i) <- Lk.zipWithIndex
          (x2, j) <- Lk.zipWithIndex
          if i > j
          if Lk(i).toArray.take(k - 2).toSet == Lk(j).toArray.take(k - 2).toSet
    } yield x1 union x2).map(ItemSet.apply)
  }

  def apriori(dataSet: Array[Array[Int]], minSupport: Double = 0.5): (Array[Array[ItemSet]], Supports) = {
    val C1 = createC1(dataSet)
    val D = dataSet.map(ItemSet.apply)
    val (l1, supportData) = scanD(D, C1, minSupport)

    @tailrec
    def loop(curL: Array[Array[ItemSet]], curSupport: Supports, curK: Int = 2): (Array[Array[ItemSet]], Supports) = {
      if (curL(curK - 2).isEmpty) (curL, curSupport)
      else {
        val Ck = aprioriGen(curL(curK - 2), curK)
        val (lk, supK) = scanD(D, Ck, minSupport)
        loop(curL :+ lk, curSupport ++ supK, curK + 1)
      }
    }
    loop(Array(l1), supportData)
  }

  def generateRules(ls: Array[Array[ItemSet]], supportData: Supports, minConf: Double = 0.7): Array[Rule] = {
    // only get the sets with two or more items
    ls.drop(1).flatMap { case items =>
      items.flatMap { freqSet =>
        if (freqSet.size <= 2) calcConf(freqSet, freqSet.toH1, supportData, minConf)
        else rulesFromConseq(freqSet, freqSet.toH1, supportData, minConf)
      }
    }
  }

  def calcConf(freqSet: ItemSet,
               H: Array[ItemSet],
               supportData: Supports,
               minConf: Double): Array[Rule] = {

    H.map { conseq =>
      (conseq, supportData(freqSet) / supportData(freqSet -- conseq))
    }.filter(_._2 >= minConf).map { case (conseq, conf) =>
      Rule(freqSet -- conseq, conseq, conf)
    }
  }

  def rulesFromConseq(freqSet: ItemSet,
                      H: Array[ItemSet],
                      supportData: Supports,
                      minConf: Double,
                      state: Array[Rule] = Array.empty): Array[Rule] = {
    val m = H(0).size
    if (freqSet.size <= m + 1) state
    else {
      val Hmp1 = state ++ calcConf(freqSet, aprioriGen(H, m + 1), supportData, minConf)
      val prunedH = Hmp1.map(_.rightSide)
      if (prunedH.size > 1) {
        Hmp1 ++ rulesFromConseq(freqSet, prunedH, supportData, minConf, Hmp1)
      } else Hmp1
    }
  }

  case class Supports(map: Map[ItemSet, Double] = Map.empty[ItemSet, Double]) {

    override def toString = s"Supports:\n${map.mkString("\n")}"

    def +(kv: (ItemSet, Double)): Supports = new Supports(map + kv)

    def ++(supp: Supports): Supports = new Supports(map ++ supp.map)

    def apply(set: ItemSet): Double = map(set)
  }

  object Supports {
    def empty = new Supports
  }

  case class ItemSet(x: Set[Int]) extends Set[Int] {

    override def toString() = s"Items[${x.mkString(", ")}]"

    def toH1: Array[ItemSet] = x.map(i => ItemSet(i)).toArray

    def contains(elem: Int): Boolean = x.contains(elem)

    def +(elem: Int): Set[Int] = x + elem

    def -(elem: Int): Set[Int] = x - elem

    def iterator: Iterator[Int] = x.iterator
  }

  object ItemSet {
    def apply(x1: Int): ItemSet = new ItemSet(Set(x1))

    def apply(arr: Array[Int]): ItemSet = new ItemSet(arr.toSet)

    implicit def set2ItemSet(x: Set[Int]): ItemSet = new ItemSet(x)
  }

  case class Rule(leftSide: ItemSet, rightSide: ItemSet, confidence: Double) {
    override def toString = f"[${leftSide.mkString(",")}] ---> [${rightSide.mkString(",")}] : confidence:$confidence"
  }

  implicit def array2ItemSet(ds: Array[Array[Int]]): Array[ItemSet] = ds.map(ItemSet.apply)
}