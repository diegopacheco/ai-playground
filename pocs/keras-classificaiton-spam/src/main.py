import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, '.')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # less verbose logs

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import pickle
from src.model import *

# Load the dataset
train_data = pd.read_csv('spam_data.csv', error_bad_lines=False)
print(train_data.columns)

# Split the data into training and testing sets
train_text, train_labels = train_data['text'], train_data['label']

test_text, test_labels = train_data['text'].head(1000), train_data['label'].head(1000)

# Create a tokenizer to split the text into words
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_text)

train_sequences = tokenizer.texts_to_sequences(train_text)
test_sequences = tokenizer.texts_to_sequences(test_text)

# Pad the sequences to have the same length
padded_train = pad_sequences(train_sequences, maxlen=max_length)
padded_test = pad_sequences(test_sequences, maxlen=max_length)

# One-hot encode the labels
label_encoder = LabelEncoder()
train_labels_encoded = label_encoder.fit_transform(train_labels)
test_labels_encoded = label_encoder.transform(test_labels)

# Convert the encoded labels to categorical variables
train_labels_categorical = to_categorical(train_labels_encoded)
test_labels_categorical = to_categorical(test_labels_encoded)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(padded_train, train_labels_categorical, epochs=10, batch_size=32, validation_data=(padded_test, test_labels_categorical))

# Evaluate the model on the testing data
loss, accuracy = model.evaluate(padded_test, test_labels_categorical)
print(f'Test loss: {loss:.3f}, Test accuracy: {accuracy:.3f}')

# Use the model to predict new text as either spam or not spam
new_text = "This is a sample spam message"
new_sequence = tokenizer.texts_to_sequences([new_text])[0]
padded_new = pad_sequences([new_sequence], maxlen=max_length)

prediction = model.predict(padded_new)
if prediction[0][1] > 0.5:
    print("Spam")
else:
    print("Not Spam")

model.save('spam_model.keras')
print("Model saved to spam_model.keras")

# Save the tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Tokenizer saved to tokenizer.pickle")    