package mlia.fpgrowth

object Prep {
                
  def loadSimpDat: Array[Array[String]] =
    Array(Array("r", "z", "h", "j", "p"),
      Array("z", "y", "x", "w", "v", "u", "t", "s"),
      Array("z"),
      Array("r", "x", "n", "o", "s"),
      Array("y", "r", "x", "z", "q", "t", "p"),
      Array("y", "z", "x", "e", "q", "s", "t", "m"))

  def createInitSet(dataSet: Array[Array[String]]): Map[Set[String], Int] = dataSet.map(arr => (arr.toSet, 1)).toMap
}
