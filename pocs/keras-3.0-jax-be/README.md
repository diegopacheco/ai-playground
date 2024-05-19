### Result
* Kera 3.0
* Jax backend

Result:
```
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 conv2d (Conv2D)             (None, 26, 26, 64)        640

 conv2d_1 (Conv2D)           (None, 24, 24, 64)        36928

 max_pooling2d (MaxPooling2D  (None, 12, 12, 64)       0
 )

 conv2d_2 (Conv2D)           (None, 10, 10, 128)       73856

 conv2d_3 (Conv2D)           (None, 8, 8, 128)         147584

 global_average_pooling2d (G  (None, 128)              0
 lobalAveragePooling2D)

 dropout (Dropout)           (None, 128)               0

 dense (Dense)               (None, 10)                1290

=================================================================
Total params: 260,298
Trainable params: 260,298
Non-trainable params: 0
_________________________________________________________________
None
Epoch 1/20
399/399 [==============================] - 105s 262ms/step - loss: 0.7370 - acc: 0.7496 - val_loss: 0.1293 - val_acc: 0.9643
Epoch 2/20
399/399 [==============================] - 96s 241ms/step - loss: 0.2127 - acc: 0.9368 - val_loss: 0.0868 - val_acc: 0.9743
Epoch 3/20
399/399 [==============================] - 108s 270ms/step - loss: 0.1566 - acc: 0.9543 - val_loss: 0.0599 - val_acc: 0.9820
Epoch 4/20
399/399 [==============================] - 120s 300ms/step - loss: 0.1209 - acc: 0.9643 - val_loss: 0.0677 - val_acc: 0.9793
Epoch 5/20
399/399 [==============================] - 110s 275ms/step - loss: 0.1079 - acc: 0.9681 - val_loss: 0.0479 - val_acc: 0.9859
Epoch 6/20
399/399 [==============================] - 109s 272ms/step - loss: 0.0924 - acc: 0.9720 - val_loss: 0.0468 - val_acc: 0.9873
Epoch 7/20
399/399 [==============================] - 106s 266ms/step - loss: 0.0827 - acc: 0.9746 - val_loss: 0.0384 - val_acc: 0.9887
Epoch 8/20
399/399 [==============================] - 127s 317ms/step - loss: 0.0726 - acc: 0.9777 - val_loss: 0.0387 - val_acc: 0.9892
Epoch 9/20
399/399 [==============================] - 107s 269ms/step - loss: 0.0720 - acc: 0.9792 - val_loss: 0.0331 - val_acc: 0.9898
Epoch 10/20
399/399 [==============================] - 147s 369ms/step - loss: 0.0619 - acc: 0.9816 - val_loss: 0.0324 - val_acc: 0.9911
Epoch 11/20
399/399 [==============================] - 156s 392ms/step - loss: 0.0580 - acc: 0.9821 - val_loss: 0.0285 - val_acc: 0.9912
Epoch 12/20
399/399 [==============================] - 123s 309ms/step - loss: 0.0519 - acc: 0.9843 - val_loss: 0.0350 - val_acc: 0.9906
Epoch 13/20
399/399 [==============================] - 105s 261ms/step - loss: 0.0530 - acc: 0.9841 - val_loss: 0.0295 - val_acc: 0.9917
Score [0.02570790983736515, 0.9919000267982483] for model ['loss', 'acc']
Model saved as final_model.keras
313/313 [==============================] - 7s 20ms/step
Predictions shape: (10000, 10)
```