package mlia.bayes

import breeze.linalg._
import breeze.numerics._

object NaiveBayes {

  case class Prob(num: Vector[Int], denom: Double) {

    def probability: Vector[Double] = num.mapValues(_.toDouble) :/ denom

    def logProbability: Vector[Double] = log(num.mapValues(_.toDouble) :/ denom)
  }

  object Prob {
    def apply(size: Int): Prob = Prob(DenseVector.ones(size), 2.0d) // avoid 0
  }

  def trainNB0(trainMatrix: DenseMatrix[Int], trainCategory: Vector[Int]): (Prob, Prob, Double) = {

    val numTrainDocs = trainMatrix.rows
    val numWords = trainMatrix.cols

    val probs = (0 until numTrainDocs).foldLeft((Prob(numWords), Prob(numWords))) { (state, i) =>
      val v: Vector[Int] = trainMatrix(i, ::).toDenseVector // [0, 1, 0, 0, 1, 0...]
      // vector addition
      if (trainCategory(i) == 1) (Prob(state._1.num + v, state._1.denom + v.sum), state._2) // add up class=1
      else (state._1, Prob(state._2.num + v, state._2.denom + v.sum))
    }
    (probs._2, probs._1, trainCategory.sum / numTrainDocs.toDouble) // probability, class=0, class=1, abusive
  }

  def classifyNB(vec2Classify: Vector[Int], p0Vec: Vector[Double], p1Vec: Vector[Double], pClass1: Double) = {
    val p1 = (vec2Classify.mapValues(_.toDouble) :* p1Vec: Vector[Double]).sum + log(pClass1)
    val p0 = (vec2Classify.mapValues(_.toDouble) :* p0Vec: Vector[Double]).sum + log(1.0 - pClass1)
    if (p1 > p0) 1 else 0
  }
}
