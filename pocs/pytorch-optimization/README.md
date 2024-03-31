### What is optimization?

* Optimization is the process of adjusting model parameters to reduce model error in each training step.
* Optimizer algos:
  * ADAM
  * RMSProp
* Optimizing its(model) parameters on our data(training data).
* Training a model is an iterative process
* Each iteration the model makes a guess about the output, calculates the error in its guess (loss)
* Loss function: measures the degree of dissimilarity of obtained result to the target value, and it is the loss function that we want to minimize during training
  * Common loss functions:
     * nn.MSELoss (Mean Square Error)       --- for regression
     * nn.NLLLoss (Negative Log Likelihood) --- for classification
     * nn.CrossEntropyLoss combines nn.LogSoftmax and nn.NLLLoss.
* Collects the derivatives of the error with respect to its parameters 
* Optimizes these parameters using gradient descent
* Back propagation explained: https://www.youtube.com/watch?v=tIeHLnjs5U8
* Hyperparameters (adjustable parameters)
 * Let you control the model optimization process
 * Different hyperparameter values can impact model training and convergence rates
 * Number of Epochs - the number times to iterate over the dataset
 * Batch Size - the number of data samples propagated through the network before the parameters are updated
 * Learning Rate - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

### Program output

See that after every round(epoch) the acuracy goes up and error goes down.
```
❯ ./run.sh
Epoch 1
-------------------------------
loss: 2.293438  [   64/60000]
loss: 2.285035  [ 6464/60000]
loss: 2.269877  [12864/60000]
loss: 2.265692  [19264/60000]
loss: 2.235628  [25664/60000]
loss: 2.207377  [32064/60000]
loss: 2.221479  [38464/60000]
loss: 2.178303  [44864/60000]
loss: 2.175148  [51264/60000]
loss: 2.148814  [57664/60000]
Test Error:
 Accuracy: 41.4%, Avg loss: 2.139438

Epoch 2
-------------------------------
loss: 2.151555  [   64/60000]
loss: 2.142899  [ 6464/60000]
loss: 2.083442  [12864/60000]
loss: 2.099650  [19264/60000]
loss: 2.043137  [25664/60000]
loss: 1.977545  [32064/60000]
loss: 2.013603  [38464/60000]
loss: 1.924887  [44864/60000]
loss: 1.930519  [51264/60000]
loss: 1.864431  [57664/60000]
Test Error:
 Accuracy: 52.7%, Avg loss: 1.857007

Epoch 3
-------------------------------
loss: 1.894248  [   64/60000]
loss: 1.868677  [ 6464/60000]
loss: 1.741137  [12864/60000]
loss: 1.786249  [19264/60000]
loss: 1.680153  [25664/60000]
loss: 1.626515  [32064/60000]
loss: 1.658125  [38464/60000]
loss: 1.554423  [44864/60000]
loss: 1.580771  [51264/60000]
loss: 1.486055  [57664/60000]
Test Error:
 Accuracy: 59.3%, Avg loss: 1.495643

Epoch 4
-------------------------------
loss: 1.563342  [   64/60000]
loss: 1.537126  [ 6464/60000]
loss: 1.377309  [12864/60000]
loss: 1.457972  [19264/60000]
loss: 1.347251  [25664/60000]
loss: 1.330353  [32064/60000]
loss: 1.356643  [38464/60000]
loss: 1.276255  [44864/60000]
loss: 1.309798  [51264/60000]
loss: 1.223147  [57664/60000]
Test Error:
 Accuracy: 63.0%, Avg loss: 1.240350

Epoch 5
-------------------------------
loss: 1.315051  [   64/60000]
loss: 1.306037  [ 6464/60000]
loss: 1.130745  [12864/60000]
loss: 1.244178  [19264/60000]
loss: 1.130553  [25664/60000]
loss: 1.136640  [32064/60000]
loss: 1.168856  [38464/60000]
loss: 1.101952  [44864/60000]
loss: 1.138783  [51264/60000]
loss: 1.065514  [57664/60000]
Test Error:
 Accuracy: 64.5%, Avg loss: 1.079111

Epoch 6
-------------------------------
loss: 1.147907  [   64/60000]
loss: 1.158412  [ 6464/60000]
loss: 0.965784  [12864/60000]
loss: 1.107179  [19264/60000]
loss: 0.995058  [25664/60000]
loss: 1.005152  [32064/60000]
loss: 1.050786  [38464/60000]
loss: 0.990168  [44864/60000]
loss: 1.027122  [51264/60000]
loss: 0.965292  [57664/60000]
Test Error:
 Accuracy: 65.6%, Avg loss: 0.974187

Epoch 7
-------------------------------
loss: 1.031336  [   64/60000]
loss: 1.062441  [ 6464/60000]
loss: 0.852749  [12864/60000]
loss: 1.015553  [19264/60000]
loss: 0.909712  [25664/60000]
loss: 0.913382  [32064/60000]
loss: 0.973274  [38464/60000]
loss: 0.917607  [44864/60000]
loss: 0.950831  [51264/60000]
loss: 0.898463  [57664/60000]
Test Error:
 Accuracy: 66.8%, Avg loss: 0.903097

Epoch 8
-------------------------------
loss: 0.945986  [   64/60000]
loss: 0.996373  [ 6464/60000]
loss: 0.772516  [12864/60000]
loss: 0.951595  [19264/60000]
loss: 0.853362  [25664/60000]
loss: 0.847269  [32064/60000]
loss: 0.918947  [38464/60000]
loss: 0.869342  [44864/60000]
loss: 0.896681  [51264/60000]
loss: 0.850561  [57664/60000]
Test Error:
 Accuracy: 68.0%, Avg loss: 0.852428

Epoch 9
-------------------------------
loss: 0.881261  [   64/60000]
loss: 0.947267  [ 6464/60000]
loss: 0.712854  [12864/60000]
loss: 0.904661  [19264/60000]
loss: 0.813353  [25664/60000]
loss: 0.798129  [32064/60000]
loss: 0.877883  [38464/60000]
loss: 0.835922  [44864/60000]
loss: 0.856312  [51264/60000]
loss: 0.814161  [57664/60000]
Test Error:
 Accuracy: 69.3%, Avg loss: 0.814229

Epoch 10
-------------------------------
loss: 0.829637  [   64/60000]
loss: 0.908125  [ 6464/60000]
loss: 0.666448  [12864/60000]
loss: 0.868730  [19264/60000]
loss: 0.782654  [25664/60000]
loss: 0.760653  [32064/60000]
loss: 0.844751  [38464/60000]
loss: 0.811053  [44864/60000]
loss: 0.824829  [51264/60000]
loss: 0.784952  [57664/60000]
Test Error:
 Accuracy: 70.6%, Avg loss: 0.783830

Done!
```