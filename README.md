# Machine Learning in Action for Scala (mlia-scala)

## k-Nearest Neighbors
    import breeze.linalg._
    import mlia.knn.KNearestNeighbors
     
    val ds: DenseMatrix[Double] = DenseMatrix((1.0, 1.0), (1.0, 2.0), (1.0, 3.0), (10.0, 11.0), (12.0, 13.0))
    val labels = Vector("A", "A", "B", "B", "B")
    val k = 3
    val inX = Vector[Double](13, 10)

    KNearestNeighbors.classify0(inX, ds, labels, k)
    
## See also
    [Data Fun!]: http://data-fun.machine-learning.cloudbees.net/