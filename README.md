# Machine Learning in Action for Scala

<a target="_blank" href="http://www.amazon.co.jp/Machine-Learning-Action-Peter-Harrington/dp/1617290181/?_encoding=UTF8&camp=247&creative=1211&linkCode=ur2&tag=noborukudoh-22">Machine Learning in Action</a><img src="http://ir-jp.amazon-adsystem.com/e/ir?t=noborukudoh-22&l=ur2&o=9" width="1" height="1" border="0" alt="" style="border:none !important; margin:0px !important;" />のサンプルコードはpythonで書かれていますが、scala版を作ってみました。
現在は主要なアルゴリズム部分のみで、グラフィック関連のコードは省略しています。

## Chapter2 : k-Nearest Neighbors(k近傍法)

```scala
import breeze.linalg._
import mlia.knn.KNearestNeighbors
     
val ds = DenseMatrix((1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (10.0, 11.0), (12.0, 13.0))
val labels = Vector("A", "A", "B", "B", "B") // actual labels of ds
val k = 3 // how many neighbor
val inX = Vector[Double](13, 10) // vector which you want to know

KNearestNeighbors.classify0(inX, ds, labels, k) // return "B"
```

## Chapter3 : Decision Trees(決定木)
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

## Chapter4 : Naive Bayes(単純ベイズ)
```scala
import breeze.linalg._
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
// 1  1  1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  ... (32 total)
// 0  1  0  0  0  0  0  1  1  1  1  1  1  1  0  0  0  0  0  0  0  ...
// 1  0  0  0  0  0  0  0  0  0  1  0  0  0  1  1  1  1  1  1  0  ...
// 0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  1  ...
// ...

// train by naive bayes
val (p0v, p1v, pAb) = trainNB0(trainMat, listClasses)

// it has a risk to get underflow and incorrect answer
println(p0v.probability)
// DenseVector(0.15384615384615385, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, ...)

println(p1v.probability)
// DenseVector(0.047619047619047616, 0.14285714285714285, 0.047619047619047616, 0.047619047619047616, 0.047619047619047616, ...)

// it's stable to calculate on the computer! ,but it's not understandable.
println(p0v.logProbability)
// DenseVector(-1.8718021769015913, -2.5649493574615367, -2.5649493574615367, -2.5649493574615367, -2.5649493574615367, ...)
println(p1v.logProbability)
// DenseVector(-3.044522437723423, -1.9459101490553135, -3.044522437723423, -3.044522437723423, -3.044522437723423, -3.044522437723423, ...)

// total abusive probability(class == 1)
println(pAb) // 0.5

// test bayes
val thisDoc0 = setOfWords2Vec(myVocabList, Array("love", "my", "dalmation"))
println(s"classified as: ${classifyNB(thisDoc0, p0v.logProbability, p1v.logProbability, pAb)}") 
// => classified as "0"

val thisDoc1 = setOfWords2Vec(myVocabList, Array("stupid", "garbage"))
println(s"classified as: ${classifyNB(thisDoc1, p0v.logProbability, p1v.logProbability, pAb)}")
// => classified as "1"

```

## Chapter5 : Logistic Regression(ロジスティック回帰)
```scala
import mlia.lr.LogisticRegression._
import mlia.lr.Prep._

val (dataMat, labelMat) = loadDataSet("/lr/testSet.txt")
// normal gradient ascent optimization. it's expensive to compute.
gradAscent(dataMat, labelMat.toArray)
// => DenseVector(4.124143489627893, 0.48007329288424455, -0.6168481970344017)

// stochastic gradient ascent. it's faster than normal gradient ascent. but it often misclassify. 
stocGradAscent0(dataMat, labelMat.toArray)
// => DenseVector(1.0170200728876158, 0.859143479425245, -0.36579921045742)

// improve stochastic gradient ascent.
stocGradAscent1(dataMat, labelMat.toArray)
// => DenseVector(13.940485981986548, 0.8592396791079943, -1.8701169404631004)

// test each algorithm by error rate
import mlia.lr.ColicTest._

calcErrorRateMean("/lr/horseColicTraining.txt","/lr/horseColicTest.txt")(gradAscent)
// the error rate of this test is: 0.283582
// the error rate of this test is: 0.283582
// ...
// after 10 iterations the average error rate is: 0.283582 <- well, not bad.

calcErrorRateMean("/lr/horseColicTraining.txt","/lr/horseColicTest.txt")(stocGradAscent0)
// the error rate of this test is: 0.507463
// the error rate of this test is: 0.507463
// ...
// after 10 iterations the average error rate is: 0.507463 <- bad performance...

calcErrorRateMean("/lr/horseColicTraining.txt","/lr/horseColicTest.txt")(stocGradAscent1Iter500)
// the error rate of this test is: 0.014925
// the error rate of this test is: 0.000000
// ...
// after 10 iterations the average error rate is: 0.026866 <- this performance is too good than the book for some reason.

```

## Chapter6 : Support Vector Machine(サポートベクターマシーン)
Simplified SMO Algorithm
```scala
import breeze.linalg._
import mlia.svm.Prep._
import mlia.svm.SimplifiedSMO._

val (dataArr, labelArr) = loadDataSet("/svm/testSet.txt")
val (alphas, b): (DenseMatrix[Double], Double) = smoSimple(dataArr.toArray, labelArr.toArray, 0.6, 0.001, 40)
// iter: 0 i:0, pairs changed 1
// iter: 0 i:2, pairs changed 2
// j not moving enough[0.0]
// iter: 0 i:8, pairs changed 3
// j not moving enough[0.0]
// ...

alphas.findAll(_ > 0.0).foreach {
  case (row, col) => println(alphas(row,col))
}
// 0.12756738739781906
// 0.2416951660901028
// 2.7755575615628914E-17
// 0.3692625534879218

println(b)
// -3.8418049116532984
```

Full SMO
```scala
// Full Platt SMO. This algorithm is more faster than Simplified SMO.
import breeze.linalg._
import mlia.svm.FullSMO._
import mlia.svm.Prep._
val (dataArr, labelArr) = loadDataSet("/svm/testSet.txt")

val (alphas, b) = smoP(dataArr.toArray, labelArr.toArray, 0.6, 0.001, 40)
// L == H[0.0]
// fullSet, iter: 0 i:0, pairs changed 0
// L == H[0.0]
// fullSet, iter: 0 i:1, pairs changed 0
...
// non-bound, iter: 1 i:10, pairs changed 2
// j not moving enough[0.0]
// non-bound, iter: 1 i:18, pairs changed 3
// j not moving enough[0.0]
...

println(b)
// -3.4003419604099356

alphas.findAll(_ > 0.0).foreach {
  case (row, col) => println(s"support vector: ${dataArr(row).mkString(",")}, alpha: ${alphas(row,col)}, label: ${labelArr(row)}")
}
// support vector: 3.542485,1.977398, alpha: 0.11718183426962622, label: -1.0
// support vector: 3.223038,-0.552392, alpha: 0.022762846775601038, label: -1.0
// support vector: 7.286357,0.251077, alpha: 0.0135202386819083, label: 1.0
// support vector: 3.457096,-0.082216, alpha: 0.0135202386819083, label: -1.0
// support vector: 2.893743,-1.643468, alpha: 0.0908201901982198, label: -1.0
// support vector: 5.286862,-2.358286, alpha: 0.11358303697382083, label: 1.0
// support vector: 6.080573,0.418886, alpha: 0.11718183426962622, label: 1.0

// now, classify dataset
val ws = calcWs(alphas, dataArr, labelArr.toArray)
// 0.5174891451517396    
// -0.10098649982626291 

val dataMat = DenseMatrix(dataArr: _*)
println(s"ws: ${(dataMat(0, ::) * ws: DenseMatrix[Double]) :+ b}, actual: ${labelArr(0)}")
// ws: -1.0258808254189797  , actual: -1.0
println(s"ws: ${(dataMat(2, ::) * ws: DenseMatrix[Double]) :+ b}, actual: ${labelArr(2)}")
// ws: 2.376492433975608  , actual: 1.0
```

Using Kernel Function
```scala
import breeze.linalg._
import mlia.svm.FullSMOWithKernel._

// radial bias function
val dataMat = DenseMatrix(dataArr: _*)
val labelMat = DenseMatrix(labelArr.toArray).t
val os = OptStruct(dataMat,labelMat, DenseMatrix.zeros[Double](dataMat.rows, 1),0.0,200,0.0001,Kernel("rbf", Array(1.3)))
println(os.k) // dataMat is transformed by kernel transformation
// 1.0                  0.7040816347352673   0.7954283622305077   ... (100 total)
// 0.7040816347352673   1.0                  0.6351840054854467   ...
// 0.7954283622305077   0.6351840054854467   1.0                  ...
// 0.7545606658622943   0.9256097040596233   0.836444104017469    ...
// ...
// 0.881210180460299    0.7585850539381088   0.9726618561420948   ...
// ... (100 total)

// training svm with radial bias function
val (alphas, b) = smoP(dataArr.toArray, labelArr.toArray, 200.0, 0.0001, 10000,Kernel("rbf", Array(1.3)))
// fullSet, iter: 0 i:0, pairs changed 0
// L == H[0.0]
// fullSet, iter: 0 i:1, pairs changed 1
// fullSet, iter: 0 i:2, pairs changed 1
// L == H[0.0]
// fullSet, iter: 0 i:3, pairs changed 1
// fullSet, iter: 0 i:4, pairs changed 1
// ...

alphas.findAll(_ > 0.0).foreach {
  case (row, col) => println(s"support vector: ${dataArr(row).mkString(",")}, alpha: ${alphas(row, col)}, label: ${labelArr(row)}")
}
// support vector: -0.214824,0.662756, alpha: 0.6, label: -1.0
// support vector: 0.22365,0.130142, alpha: 0.6, label: 1.0
// support vector: -0.7488,-0.531637, alpha: 0.6, label: -1.0
// support vector: 0.207123,-0.019463, alpha: 0.6, label: 1.0
// support vector: 0.286462,0.71947, alpha: 0.6, label: -1.0

println(b)
// 1.1356280979492999

// evaluate with test data
import mlia.svm.NonLinearTest._
calcErrorRate("/svm/testSetRBF.txt", "/svm/testSetRBF2.txt", 1.3)
// there are 20 Support Vectors
// the training error rate is: 0.00000
// the test error rate is: 0.08000

```

## Chapter7 : AdaBoost
```scala
import mlia.adaboost.DecisionStump._
import mlia.adaboost.Prep._
import breeze.linalg._

val (dataMat, classLabels) = loadSimpData()

val D = DenseMatrix.ones[Double](5, 1) :/ 5.0
val stump = buildStump(dataMat, classLabels, D)
// split: dim 0, thresh 1.00, thresh ineqal: lt, the weighted error is 0.400
// split: dim 0, thresh 1.00, thresh ineqal: gt, the weighted error is 0.600
// split: dim 0, thresh 1.10, thresh ineqal: lt, the weighted error is 0.400
// split: dim 0, thresh 1.10, thresh ineqal: gt, the weighted error is 0.600
// ...

println(stump)
// dim:0, threshold:1.3, ineqal:lt, minErr:0.2, 
// bestClassEst:
// -1.0  
// 1.0   
// -1.0  
// -1.0  
// 1.0   


```

## See also
  [Data Fun!](http://data-fun.machine-learning.cloudbees.net)
