### Training
```
❯ time ./run.sh
Step 1/4 Data Preparation
Step 2/4 Model Creation
Step 3/4 Model Training
Number of classes: 4
Epoch: 1/5, Batch: 1/1875, Loss: 1.9081844091415405, Accuracy: 26.5625%
Epoch: 1/5, Batch: 501/1875, Loss: 1.0753597021102905, Accuracy: 49.76921157684631%
Epoch: 1/5, Batch: 1001/1875, Loss: 1.1299575567245483, Accuracy: 49.73932317682318%
Epoch: 1/5, Batch: 1501/1875, Loss: 1.02041757106781, Accuracy: 49.829280479680214%
Epoch: 2/5, Batch: 1/1875, Loss: 1.0071884393692017, Accuracy: 57.8125%
Epoch: 2/5, Batch: 501/1875, Loss: 0.9698296785354614, Accuracy: 49.946981037924154%
Epoch: 2/5, Batch: 1001/1875, Loss: 1.1576697826385498, Accuracy: 49.928196803196805%
Epoch: 2/5, Batch: 1501/1875, Loss: 1.1385365724563599, Accuracy: 49.89173884077282%
Epoch: 3/5, Batch: 1/1875, Loss: 1.0773110389709473, Accuracy: 48.4375%
Epoch: 3/5, Batch: 501/1875, Loss: 1.0382325649261475, Accuracy: 50.02183133732535%
Epoch: 3/5, Batch: 1001/1875, Loss: 1.0393192768096924, Accuracy: 50.04526723276723%
Epoch: 3/5, Batch: 1501/1875, Loss: 1.0877199172973633, Accuracy: 50.042679880079945%
Epoch: 4/5, Batch: 1/1875, Loss: 1.0757807493209839, Accuracy: 48.4375%
Epoch: 4/5, Batch: 501/1875, Loss: 1.1263147592544556, Accuracy: 50.00935628742515%
Epoch: 4/5, Batch: 1001/1875, Loss: 1.0589148998260498, Accuracy: 50.02185314685315%
Epoch: 4/5, Batch: 1501/1875, Loss: 1.1070531606674194, Accuracy: 49.93962358427715%
Epoch: 5/5, Batch: 1/1875, Loss: 0.9667549729347229, Accuracy: 62.5%
Epoch: 5/5, Batch: 501/1875, Loss: 1.1583975553512573, Accuracy: 49.750499001996005%
Epoch: 5/5, Batch: 1001/1875, Loss: 1.052998423576355, Accuracy: 49.854832667332666%
Epoch: 5/5, Batch: 1501/1875, Loss: 1.0677454471588135, Accuracy: 49.87820619586942%
Total Accuracy: 49.94716666666667%
Model saved!
Step 4/4 Serving Predictions with the model
Win a free trip to Paris! Result: SPAM
Hello, how are you? Result: NOT SPAM
Congratulations, you've won a $1000 gift card! Result: NOT SPAM
./run.sh  326,29s user 1,28s system 509% cpu 1:04,28 total
```
### Predict
```
❯ time ./predict.sh
Win a free trip to Paris! Result: SPAM
Hello, how are you? Result: NOT SPAM
Congratulations, you've won a $1000 gift card! Result: NOT SPAM
./predict.sh  16,18s user 0,85s system 101% cpu 16,704 total
```
### Tuning
The loss value being high or low is relative and depends on various factors such as the complexity of your model, the difficulty of your task, the quality of your data, and the specific loss function you're using.

However, in general, a loss value above 1.0 for a classification task might indicate that your model is having difficulty learning from the data. Here are a few suggestions to improve the model's performance:

Increase Model Complexity: If your model is too simple, it might not be able to learn complex patterns in the data. Try adding more layers or increasing the number of neurons in the existing layers.

Change Learning Rate: If your learning rate is too high, the model might be overshooting the optimal solution. If it's too low, the model might be learning too slowly. Try adjusting the learning rate.

Use a Different Optimizer: Different optimizers can lead to different results. If you're using SGD, you might want to try Adam or RMSprop, which are more adaptive.

Increase Training Time: Sometimes, the model just needs more time to learn. Try increasing the number of epochs.

Regularization: If your model is overfitting, try adding some form of regularization like dropout or weight decay.

Data Augmentation: If you don't have a lot of data, data augmentation can create more training examples and help improve the model's performance.

Remember to monitor both training and validation loss during training. If your training loss is decreasing but validation loss is not, it might be a sign of overfitting. In this case, you should try techniques like regularization or early stopping.