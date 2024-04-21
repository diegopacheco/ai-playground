from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input
from keras.layers import Flatten

# Create the neural network model
max_length = 200
input_shape = (max_length, )

model = Sequential()
model.add(Input(shape=input_shape))
model.add(Embedding(input_dim=5000, output_dim=5000))
model.add(Flatten())  # Add this line
model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='softmax'))
