import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import imdb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

# Load the IMDB dataset
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# Convert the list of words to sentences
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
train_sentences = [" ".join([reverse_word_index.get(i - 3, '?') for i in train_data[j]]) for j in range(len(train_data))]
test_sentences = [" ".join([reverse_word_index.get(i - 3, '?') for i in test_data[j]]) for j in range(len(test_data))]

# Vectorize the sentences
vectorizer = CountVectorizer(binary=True)
train_vectors = vectorizer.fit_transform(train_sentences)
test_vectors = vectorizer.transform(test_sentences)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(train_vectors, train_labels)

# Test the model
predictions = model.predict(test_vectors)

# Print the accuracy
print("Accuracy:", accuracy_score(test_labels, predictions))

# Plot the confusion matrix
cm = confusion_matrix(test_labels, predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()