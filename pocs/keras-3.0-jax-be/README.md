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

```