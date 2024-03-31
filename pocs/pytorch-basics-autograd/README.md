### What is autograd?

* To train a neural network(nn) the most common algo is back propagation.
* Back propagation parameters(model weight) are adjusted acording a gradient of loss function to respect to a spesific parameter
* To compute gradients pytorch has a bulti-in differenciation engine called autograd.
* Autograd support automatic computation of gradient for any computational graph.
* Conceptually, autograd keeps a record of data (tensors) and all executed operations (along with the resulting new tensors) in a directed acyclic graph (DAG). 