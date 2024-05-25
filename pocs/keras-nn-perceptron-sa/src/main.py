import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the dataset
df = pd.read_csv('sentiment_data.csv')

# Split the data into training and testing sets
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the training data
X_train = vectorizer.fit_transform(X_train).toarray()

# Get the number of features
num_features = len(vectorizer.get_feature_names_out())

# Transform the testing data
X_test = vectorizer.transform(X_test).toarray()

# Convert the labels to numerical values
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# Create the Perceptron model
model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(num_features,), name='input_layer'))  # Update the input_shape
model.add(Dense(64, activation='relu', name='hidden_layer'))  # Add a new hidden layer
model.add(Dense(1, activation='sigmoid', name='output_layer'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Define a predict function
def predict(text):
    # Transform the text to the same format as the training data
    text = vectorizer.transform([text]).toarray()
    
    # Make the prediction
    return model.predict(text)

def plot_results():
    y_pred = model.predict(X_test)
    
    # Convert predicted values to binary
    y_pred_binary = [1 if pred > 0.5 else 0 for pred in y_pred]

    # Create color arrays
    actual_colors = ['blue' for _ in range(len(y_test))]
    pred_colors = ['red' for _ in range(len(y_pred_binary))]
    overlap_colors = ['orange' if y_test[i] == y_pred_binary[i] else 'white' for i in range(len(y_test))]

    plt.scatter(range(len(y_test)), y_test, color=actual_colors, label='Actual Sentiment')
    plt.scatter(range(len(y_pred_binary)), y_pred_binary, color=pred_colors, label='Predicted Sentiment')
    plt.scatter(range(len(y_test)), y_test, color=overlap_colors, label='Overlap')
    plt.xlabel('Index')
    plt.ylabel('Sentiment')
    plt.legend()
    plt.show()

# Call the predict function
print(predict('I love this product!'))

# Plot the results
plot_results()