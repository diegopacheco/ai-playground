### Model
* The model is a simple neural network
* The mode works as follows:
  * The input layer takes the input shape.
  * The embedding layer converts the input into dense vectors of fixed size.
  * The flatten layer flattens the input.
  * The first dense layer has 64 units and uses the ReLU activation function.
  * The second dense layer has 2 units and uses the softmax activation function.
```
model = Sequential()
model.add(Input(shape=input_shape))
model.add(Embedding(input_dim=5000, output_dim=5000))
model.add(Flatten())  # Add this line
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
```

### Training
```
  train_data = pd.read_csv('spam_data.csv', error_bad_lines=False)
Index(['text', 'label'], dtype='object')
Epoch 1/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step - accuracy: 0.5455 - loss: 0.6852 - val_accuracy: 0.5455 - val_loss: 39.0358
Epoch 2/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 784ms/step - accuracy: 0.5455 - loss: 39.0358 - val_accuracy: 1.0000 - val_loss: 0.1785
Epoch 3/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 1s/step - accuracy: 1.0000 - loss: 0.1785 - val_accuracy: 0.5455 - val_loss: 14.5982
Epoch 4/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step - accuracy: 0.5455 - loss: 14.5982 - val_accuracy: 0.4545 - val_loss: 28.7975
Epoch 5/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 716ms/step - accuracy: 0.4545 - loss: 28.7975 - val_accuracy: 0.7727 - val_loss: 1.7462
Epoch 6/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 777ms/step - accuracy: 0.7727 - loss: 1.7462 - val_accuracy: 0.5455 - val_loss: 6.2943
Epoch 7/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 797ms/step - accuracy: 0.5455 - loss: 6.2943 - val_accuracy: 0.8182 - val_loss: 0.5577
Epoch 8/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 798ms/step - accuracy: 0.8182 - loss: 0.5577 - val_accuracy: 0.5909 - val_loss: 2.5897
Epoch 9/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 774ms/step - accuracy: 0.5909 - loss: 2.5897 - val_accuracy: 1.0000 - val_loss: 4.7142e-07
Epoch 10/10
1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 749ms/step - accuracy: 1.0000 - loss: 4.7142e-07 - val_accuracy: 0.9091 - val_loss: 0.5869
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 61ms/step - accuracy: 0.9091 - loss: 0.5869
Test loss: 0.587, Test accuracy: 0.909
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 65ms/step
Spam
Model saved to spam_model.keras
Tokenizer saved to tokenizer.pickle
```

### Preditions
```
❯ time ./predict.sh
```
```
Win a trip to Hawaii this summer! Predition: Spam
John I cannot go to the party. Predition:Not Spam
Honey I'm running late, please start cooking the dinner. Not Spam
```
```
./predict.sh  14,96s user 7,16s system 141% cpu 15,663 total
```