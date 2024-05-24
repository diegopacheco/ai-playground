import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt

# Let's assume we have encoded our sentences into sequences of numbers
# and our sentiments into a sequence of classes (0 for sad, 1 for happy)
sentences = np.array([[1, 2, 3, 4, 5], [2, 3, 4, 5, 6], [3, 4, 5, 6, 7]]).T
sentiments = np.array([0, 1, 1, 0, 1])

# Create a Gaussian HMM
model = hmm.GaussianHMM(n_components=2)

# Train the HMM
model.fit(sentences)

# Predict the sentiment for a new sentence
new_sentence = np.array([[4, 5, 6, 7, 8]]).T
predicted_sentiment = model.predict(new_sentence)

# Create a mapping from numbers to sentiments
sentiment_mapping = {0: 'sad', 1: 'happy'}

# Apply the mapping to the predicted sentiment for display
display_sentiment = [sentiment_mapping[s] for s in predicted_sentiment]
print(f"Predicted sentiment: {display_sentiment}")

# Plot the predicted sentiment using the numerical values
plt.plot(predicted_sentiment)
plt.ylabel('Sentiment')
plt.show()