import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder

# Load the data
df = pd.read_csv('purchase_history.csv')

# Preprocess the data
# Convert the events into one-hot encoded format
encoder = OneHotEncoder()
encoded_events = encoder.fit_transform(df['ProductName'].values.reshape(-1, 1)).toarray()

# Train the K-Means model
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(encoded_events)

# Assign each user to a cluster
df['Cluster'] = kmeans.predict(encoded_events)

# Save the results
df.to_csv('user_clusters.csv', index=False)

# Plot the clusters using t-SNE
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(encoded_events)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Cluster'], cmap='viridis')
plt.colorbar()
plt.show()
