import pandas as pd
import numpy as np

# Generate 100 random text samples
text_data = []
for i in range(100):
    text = ''
    if np.random.rand() < 0.5:
        # Negative sentiment
        text += 'I '
        if np.random.rand() < 0.5:
            text += 'hate '
        else:
            text += 'dislike '
        text += 'this product!'
    else:
        # Positive sentiment
        text += 'I '
        if np.random.rand() < 0.5:
            text += 'love '
        else:
            text += 'like '
        text += 'this product!'
    text_data.append(text)

# Generate corresponding sentiment labels (0 for negative, 1 for positive)
sentiment_labels = [0 if 'hate' in text or 'dislike' in text else 1 for text in text_data]

# Create a DataFrame and save to sentiment_data.csv
df = pd.DataFrame({'text': text_data, 'sentiment': sentiment_labels})
df.to_csv('sentiment_data.csv', index=False)
