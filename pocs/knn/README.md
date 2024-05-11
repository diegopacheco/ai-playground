### Result
* K nearest neighbors
* KNN works by finding the distances between a query and all the examples in the data, selecting the specified number examples (K) closest to the query, then votes for the most frequent label (in the case of classification) or averages the labels (in the case of regression).
* KNN is a type of instance-based learning, or lazy learning, where the function is only approximated locally and all computation is deferred until function evaluation.
* The KNN algorithm is among the simplest of all machine learning algorithms.

Resuklt:
```
X training data: [[0.66197375 0.73396028 0.8662156 ]
 [0.394334   0.13642738 0.80650251]
 [0.91531146 0.3335666  0.81288577]
 [0.39693755 0.63103087 0.36131821]
 [0.17795758 0.75907087 0.98005196]
 [0.39927565 0.84941965 0.37092551]
 [0.83871945 0.17775176 0.49306617]
 [0.1071512  0.59490452 0.18582895]
 [0.06260451 0.57024115 0.45118084]
 [0.23344708 0.7714073  0.91010654]] Y training data: [1 1 0 1 1 0 0 1 0 0 1 0 0 1 1 1 0 1 0 0 0 0 0 1 1 1 0 0 0 1] - X is [0.42502763 0.99587081 0.79695262]
Prediction: 1
```