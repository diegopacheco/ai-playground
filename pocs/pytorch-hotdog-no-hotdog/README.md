### Training
```
./run.sh
```
```
Epoch 1/10.. Train loss: 2.019.. Test loss: 1.351.. Test accuracy: 0.710
Epoch 2/10.. Train loss: 0.485.. Test loss: 2.022.. Test accuracy: 0.705
Epoch 3/10.. Train loss: 0.427.. Test loss: 1.530.. Test accuracy: 0.744
Epoch 4/10.. Train loss: 0.446.. Test loss: 1.391.. Test accuracy: 0.756
Epoch 5/10.. Train loss: 0.360.. Test loss: 1.486.. Test accuracy: 0.760
Epoch 6/10.. Train loss: 0.416.. Test loss: 1.377.. Test accuracy: 0.760
Epoch 7/10.. Train loss: 0.405.. Test loss: 1.473.. Test accuracy: 0.759
Epoch 8/10.. Train loss: 0.344.. Test loss: 2.283.. Test accuracy: 0.710
Epoch 9/10.. Train loss: 0.410.. Test loss: 1.780.. Test accuracy: 0.722
Epoch 10/10.. Train loss: 0.269.. Test loss: 1.548.. Test accuracy: 0.767
Model saved!
```
### Predict
```
./predict.sh
```
```
data/train/hotdog/106.jpg             -> Hotdog
data/train/hotdog/120.jpg             -> Hotdog
data/train/nothotdog/101.jpg          -> Not hotdog
data/predict-test/blue-car.jpg        -> Not hotdog
data/predict-test/hotdog.jpg          -> Hotdog
data/predict-test/doudle-hotdog.jpg   -> Hotdog
data/predict-test/pet-dog.jpg         -> Not hotdog
data/predict-test/pet-dog2.jpg        -> Not hotdog
```