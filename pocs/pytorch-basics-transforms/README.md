### what is Transforms?

* Daatasets not always are in the shape we need.
* we use Transforms to make manipulations on the data.
* PyTorch datasets have: transfom and target_transform
  * modify labels
  * lable need to accept callables
* ToTensor() converns a PIL image into FloatTensor.  