package mlia.bayes

import breeze.linalg._

object Prep {

  def loadDataSet: (Array[Array[String]], Vector[Int]) = {

    val postingList = Array(
      Array("my", "dog", "has", "flea", "problems", "help", "please"),
      Array("maybe", "not", "take", "him", "to", "dog", "park", "stupid"),
      Array("my", "dalmation", "is", "so", "cute", "I", "love", "him"),
      Array("stop", "posting", "stupid", "worthless", "garbage"),
      Array("mr", "licks", "ate", "my", "steak", "how", "to", "stop", "him"),
      Array("quit", "buying", "worthless", "dog", "food", "stupid"))

    val classVec = DenseVector(0, 1, 0, 1, 0, 1)

    (postingList, classVec)
  }

  def createVocabList(dataSet: Array[Array[String]]): Array[String] = dataSet.flatten.distinct

  def setOfWords2Vec(vocabList: Array[String], inputSet: Array[String]): DenseVector[Int] = {
    val returnVec: DenseVector[Int] = DenseVector.zeros[Int](vocabList.size)
    inputSet.foreach { word =>
      if (vocabList.contains(word)) returnVec(vocabList.indexOf(word)) = 1
      else println(s"the word: $word is not in my Vocabulary!")
    }
    returnVec
  }
}
