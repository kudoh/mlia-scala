package mlia.adaboost

import breeze.linalg._

object Prep {
  def loadSimpData(): (Array[Array[Double]], Array[Double]) =
    (Array(Array(1.0, 2.1), Array(2.0, 1.1), Array(1.3, 1.0), Array(1.0, 1.0), Array(2.0, 1.0)),
      Array(1.0, 1.0, -1.0, -1.0, 1.0))
}
