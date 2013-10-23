# Machine Learning in Action for Scala
translate sample code from python to scala.

http://www.amazon.co.jp/Machine-Learning-Action-Peter-Harrington/dp/1617290181

## k-Nearest Neighbors

```scala
import breeze.linalg._
import mlia.knn.KNearestNeighbors
     
val ds = DenseMatrix((1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (10.0, 11.0), (12.0, 13.0))
val labels = Vector("A", "A", "B", "B", "B") // actual labels of ds
val k = 3 // how many neighbor
val inX = Vector[Double](13, 10) // vector which you want to know

KNearestNeighbors.classify0(inX, ds, labels, k) // return "B"
```

## Decision Trees
```scala
import mlia.trees.Tree._

val ds = Array(
  Row(Array(1, 1), "yes"),
  Row(Array(1, 1), "yes"),
  Row(Array(1, 0), "no"),
  Row(Array(0, 1), "no"),
  Row(Array(0, 1), "no"))

val labels = Array("no surfacing", "flippers")
calcShannonEnt(ds)           // 0.9709505944546686
chooseBestFeatureToSplit(ds) // InformationGain(0,0.4199730940219749)
val tree = Tree(ds, labels)
// Tree[{no surfacing : 1 {flippers : 1  -> yes[Leaf]},{flippers : 0  -> no[Leaf]}},{no surfacing : 0  -> no[Leaf]}]
tree.classify(Vector(1, 0), labels) // classified as "no"
tree.classify(Vector(1, 1), labels) // classified as "yes"
```

## Naive Bayes
```scala
import breeze.linalg.DenseMatrix
import mlia.bayes.Prep._
import mlia.bayes.NaiveBayes._

val (listOPosts, listClasses) = loadDataSet
// listOPosts => Array(Array(my, dog, has, flea, problems, help, please), Array(maybe, not, take, him, to, dog, park, stupid), ...
// listClasses => DenseVector(0, 1, 0, 1, 0, 1)
val myVocabList = createVocabList(listOPosts)
// Array(my, dog, has, flea, problems, help, please, maybe, not, take, him, to, .....

// transform into a word vector
val trainMat = DenseMatrix.zeros[Int](listOPosts.size, myVocabList.size)
listOPosts.zipWithIndex.foreach(post => trainMat(post._2, ::) := setOfWords2Vec(myVocabList, post._1))
println(trainMat)
1  1  1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  ... (32 total)
0  1  0  0  0  0  0  1  1  1  1  1  1  1  0  0  0  0  0  0  0  ...
1  0  0  0  0  0  0  0  0  0  1  0  0  0  1  1  1  1  1  1  0  ...
0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  1  ...
...

// train by naive bayes
val (p0v, p1v, pAb) = trainNB0(trainMat, listClasses)
println(p0v.probability)
// DenseVector(0.125, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, 0.041666666666666664, ...
println(p1v.probability)
// DenseVector(0.0, 0.10526315789473684, 0.0, 0.0, 0.0, 0.0, 0.0, 0.05263157894736842, 0.05263157894736842, 0.05263157894736842, 0.05263157894736842, ...
println(pAb) // 0.5
```

## See also
  [Data Fun!](http://data-fun.machine-learning.cloudbees.net)
