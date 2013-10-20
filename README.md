# Machine Learning in Action for Scala
translate sample code from python to scala.

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

## See also
  [Data Fun!](http://data-fun.machine-learning.cloudbees.net)
